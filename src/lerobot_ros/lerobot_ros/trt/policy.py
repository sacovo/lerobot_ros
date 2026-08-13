import json
import os
import torch
from .engine import TRTEngineRunner


def _check_engine_fingerprint(engine_path: str, model):
    """Refuse an engine that was exported from a different model than `model`.

    EpisodeTracker folds its normalization buffers into the traced graph, so an
    engine built before those buffers existed (or from another checkpoint) is
    numerically wrong while looking entirely healthy: same input names, same
    shapes, and the Python-side bin_centers/window come from the eager model.
    Nothing downstream would notice. Engines predating the sidecar only warn,
    since the check cannot tell "old export" from "wrong export" without it.
    """
    expected = getattr(model, "graph_fingerprint", None)
    if expected is None:
        return

    path = f"{os.path.splitext(engine_path)[0]}.fingerprint.json"
    if not os.path.exists(path):
        print(
            f"WARNING: no fingerprint beside {engine_path}; cannot verify it was "
            f"built from this checkpoint. Re-export to enable the check."
        )
        return

    with open(path) as fh:
        recorded = json.load(fh)

    actual = expected()
    if recorded.get("sha256") != actual["sha256"]:
        differing = [
            key
            for key in actual
            if key != "sha256" and recorded.get(key) != actual[key]
        ]
        raise ValueError(
            f"TRT engine {engine_path} was exported from a different model than "
            f"the loaded checkpoint (fingerprint {recorded.get('sha256')} != "
            f"{actual['sha256']}; differing: {differing or 'unknown'}). Rebuild "
            f"the engine from this checkpoint."
        )

def load_trt_policy(original_policy, engine_dir, device="cuda"):
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    try:
        from lerobot_policy_smolvla_rl.modeling_smolvla_recap import SmolVLARECAPPolicy
        HAS_RECAP = True
    except ImportError:
        HAS_RECAP = False

    if isinstance(original_policy, ACTPolicy):
        return ACTTRTPolicy(os.path.join(engine_dir, "act.trt"))
    elif HAS_RECAP and isinstance(original_policy, SmolVLARECAPPolicy):
        snapflow = original_policy.model.config.snapflow_enabled
        if snapflow:
            return RECAPSnapflowTRTPolicy(engine_dir, original_policy)
        else:
            return RECAPTRTPolicy(engine_dir, original_policy)
    elif isinstance(original_policy, SmolVLAPolicy):
        return SmolVLATRTPolicy(engine_dir, original_policy.config)
    else:
        raise ValueError(f"No TRT wrapper for {type(original_policy)}")

def pad_or_truncate(tensor, max_len, pad_value):
    curr_len = tensor.shape[1]
    if curr_len < max_len:
        pad_width = max_len - curr_len
        pad_shape = list(tensor.shape)
        pad_shape[1] = pad_width
        padding = torch.full(pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device)
        return torch.cat([tensor, padding], dim=1)
    else:
        return tensor[:, :max_len]

class ACTTRTPolicy:
    def __init__(self, engine_path: str):
        print(f"Loading ACT TRT engine from {engine_path}...")
        self.runner = TRTEngineRunner(engine_path)

    def predict_action_chunk(self, observation: dict) -> torch.Tensor:
        inputs = {}
        for name in self.runner.inputs.keys():
            if name in observation:
                inputs[name] = observation[name]
            else:
                raise ValueError(f"Expected observation input '{name}' not found in observation keys: {list(observation.keys())}")

        outputs = self.runner.run(inputs)
        return outputs["action_chunk"]

class EpisodeTrackerTRTPolicy:
    """TRT-accelerated drop-in for episode_tracker.EpisodeTracker at inference time.

    Only wraps the exported forward pass (backbone + GRU + head -> per-bin
    logits); progress_keys/bin_centers are copied from the original model
    since the HL-Gauss softmax + weighted-sum post-processing stays in Python
    (mirrors EpisodeTracker.predict_progress exactly, just calling the TRT
    engine instead of the eager forward()).
    """
    def __init__(self, engine_path: str, original_model):
        print(f"Loading EpisodeTracker TRT engine from {engine_path}...")
        _check_engine_fingerprint(engine_path, original_model)
        self.runner = TRTEngineRunner(engine_path)
        self.progress_keys = original_model.progress_keys
        self.bin_centers = original_model.bin_centers
        self.window = original_model.window

    def forward(self, batch: dict) -> torch.Tensor:
        inputs = {}
        for name in self.progress_keys:
            if name not in batch:
                raise ValueError(f"Expected progress input '{name}' not found in batch keys: {list(batch.keys())}")
            inputs[name] = batch[name]

        outputs = self.runner.run(inputs)
        return outputs["logits"]

    def predict_progress(self, batch: dict) -> torch.Tensor:
        logits = self.forward(batch)
        probs = torch.softmax(logits, dim=-1)
        bin_centers = self.bin_centers.to(probs.device)
        return (probs * bin_centers.unsqueeze(0)).sum(dim=-1)

class SmolVLATRTPolicy:
    def __init__(self, engine_dir: str, config):
        self.config = config
        self.prefix_runner = TRTEngineRunner(os.path.join(engine_dir, "smolvla_prefix.trt"))
        self.suffix_runner = TRTEngineRunner(os.path.join(engine_dir, "smolvla_suffix.trt"))
        self.num_steps = config.num_steps

    def predict_action_chunk(self, observation: dict, noise: torch.Tensor = None) -> torch.Tensor:
        device = torch.device("cuda")
        image_keys = sorted([k for k in observation.keys() if k.startswith("observation.images")])
        
        # 1. Run prefix engine
        prefix_inputs = {}
        for idx, k in enumerate(image_keys):
            prefix_inputs[self.prefix_runner.engine.get_tensor_name(idx)] = observation[k]
        
        state = observation["observation.state"]
        if state.ndim > 2:
            state = state[:, -1, :]
        prefix_inputs["state"] = pad_or_truncate(state, self.config.max_state_dim, 0.0)
        
        # Pad language inputs to static shape of 64
        lang_tokens = observation["observation.language.tokens"]
        lang_masks = observation["observation.language.attention_mask"]
        
        pad_token_id = getattr(self.config, "pad_token_id", 1)
        if pad_token_id is None:
            pad_token_id = 1
            
        lang_tokens = pad_or_truncate(lang_tokens, 64, pad_token_id)
        lang_masks = pad_or_truncate(lang_masks, 64, False)
        
        prefix_inputs["lang_tokens"] = lang_tokens
        prefix_inputs["lang_masks"] = lang_masks
        
        prefix_outputs = self.prefix_runner.run(prefix_inputs)
        
        # Reconstruct flat KV cache and prefix_pad_masks
        prefix_pad_masks = prefix_outputs["prefix_pad_masks"]
        
        # 2. Denoising loop
        bsize = observation["observation.state"].shape[0]
        if noise is None:
            noise = torch.randn(bsize, self.config.chunk_size, self.config.max_action_dim, device=device)
            
        dt = -1.0 / self.num_steps
        x_t = noise
        
        suffix_inputs = {}
        suffix_inputs["prefix_pad_masks"] = prefix_pad_masks
        for name in prefix_outputs.keys():
            if name != "prefix_pad_masks":
                suffix_inputs[name] = prefix_outputs[name]
                
        for step in range(self.num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device).expand(bsize)
            
            suffix_inputs["x_t"] = x_t
            suffix_inputs["timestep"] = t_tensor
            
            suffix_outputs = self.suffix_runner.run(suffix_inputs)
            v_t = suffix_outputs["v_t"]
            
            x_t = x_t + dt * v_t
            
        actions = x_t
        # Postprocess actions: unpad and adapt to pi aloha
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]
        
        # Aloha gripper scaling
        if getattr(self.config, "adapt_to_pi_aloha", False):
            from lerobot.policies.smolvla.modeling_smolvla import aloha_gripper_from_angular
            for motor_idx in [1, 2, 8, 9]:
                actions[:, :, motor_idx] *= -1
            for motor_idx in [6, 13]:
                actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
                
        return actions

class RECAPSnapflowTRTPolicy:
    def __init__(self, engine_dir: str, original_policy):
        self.config = original_policy.config
        self.prefix_runner = TRTEngineRunner(os.path.join(engine_dir, "smolvla_prefix.trt"))
        self.suffix_runner = TRTEngineRunner(os.path.join(engine_dir, "smolvla_suffix.trt"))
        self.use_adv = getattr(self.config, "use_advantage_conditioning", False)
        self.adv_pos_id = getattr(original_policy.model, "adv_pos_id", None)
        self.cfg_weight = getattr(self.config, "cfg_weight", 1.0)

    def predict_action_chunk(self, observation: dict, noise: torch.Tensor = None) -> torch.Tensor:
        device = torch.device("cuda")
        image_keys = sorted([k for k in observation.keys() if k.startswith("observation.images")])
        
        # 1. Run prefix engine
        prefix_inputs = {}
        for idx, k in enumerate(image_keys):
            prefix_inputs[self.prefix_runner.engine.get_tensor_name(idx)] = observation[k]
        
        state = observation["observation.state"]
        if state.ndim > 2:
            state = state[:, -1, :]
        prefix_inputs["state"] = pad_or_truncate(state, self.config.max_state_dim, 0.0)
        
        # Pad language inputs to static shape of 64
        lang_tokens = observation["observation.language.tokens"]
        lang_masks = observation["observation.language.attention_mask"]
        
        pad_token_id = getattr(self.config, "pad_token_id", 1)
        if pad_token_id is None:
            pad_token_id = 1
            
        lang_tokens = pad_or_truncate(lang_tokens, 64, pad_token_id)
        lang_masks = pad_or_truncate(lang_masks, 64, False)
        
        bsize = observation["observation.state"].shape[0]
        use_cfg = self.use_adv and self.cfg_weight != 1.0 and self.cfg_weight != 0.0
        
        if use_cfg:
            # Run unconditional prefix pass
            adv_tokens_uncond = torch.zeros((bsize, 1), dtype=torch.long, device=device)
            lang_tokens_uncond = torch.cat([lang_tokens, adv_tokens_uncond], dim=1)
            lang_masks_uncond = torch.cat([lang_masks, torch.zeros_like(adv_tokens_uncond, dtype=torch.bool)], dim=1)
            
            prefix_inputs_uncond = prefix_inputs.copy()
            prefix_inputs_uncond["lang_tokens"] = lang_tokens_uncond
            prefix_inputs_uncond["lang_masks"] = lang_masks_uncond
            prefix_outputs_uncond = self.prefix_runner.run(prefix_inputs_uncond)
            
            # Run conditional prefix pass
            adv_tokens_cond = torch.full((bsize, 1), self.adv_pos_id, dtype=torch.long, device=device)
            lang_tokens_cond = torch.cat([lang_tokens, adv_tokens_cond], dim=1)
            lang_masks_cond = torch.cat([lang_masks, torch.ones_like(adv_tokens_cond, dtype=torch.bool)], dim=1)
            
            prefix_inputs_cond = prefix_inputs.copy()
            prefix_inputs_cond["lang_tokens"] = lang_tokens_cond
            prefix_inputs_cond["lang_masks"] = lang_masks_cond
            prefix_outputs_cond = self.prefix_runner.run(prefix_inputs_cond)
        else:
            if self.use_adv and self.adv_pos_id is not None:
                adv_tokens = torch.full((bsize, 1), self.adv_pos_id, dtype=torch.long, device=device)
                lang_tokens = torch.cat([lang_tokens, adv_tokens], dim=1)
                lang_masks = torch.cat([lang_masks, torch.ones_like(adv_tokens, dtype=torch.bool)], dim=1)
            
            prefix_inputs["lang_tokens"] = lang_tokens
            prefix_inputs["lang_masks"] = lang_masks
            prefix_outputs = self.prefix_runner.run(prefix_inputs)
            
        # 2. SnapFlow step: t=1.0 to s=0.0
        if noise is None:
            noise = torch.randn(bsize, self.config.chunk_size, self.config.max_action_dim, device=device)
            
        t_tensor = torch.tensor([1.0], dtype=torch.float32, device=device).expand(bsize)
        s_tensor = torch.tensor([0.0], dtype=torch.float32, device=device).expand(bsize)
        
        if use_cfg:
            # Suffix uncond
            suffix_inputs_uncond = {"x_t": noise, "timestep": t_tensor, "target_time": s_tensor, "prefix_pad_masks": prefix_outputs_uncond["prefix_pad_masks"]}
            for name in prefix_outputs_uncond.keys():
                if name != "prefix_pad_masks":
                    suffix_inputs_uncond[name] = prefix_outputs_uncond[name]
            v_t_uncond = self.suffix_runner.run(suffix_inputs_uncond)["v_t"]
            
            # Suffix cond
            suffix_inputs_cond = {"x_t": noise, "timestep": t_tensor, "target_time": s_tensor, "prefix_pad_masks": prefix_outputs_cond["prefix_pad_masks"]}
            for name in prefix_outputs_cond.keys():
                if name != "prefix_pad_masks":
                    suffix_inputs_cond[name] = prefix_outputs_cond[name]
            v_t_cond = self.suffix_runner.run(suffix_inputs_cond)["v_t"]
            
            # CFG Blend: uncond + w * (cond - uncond)
            v_t = v_t_uncond + self.cfg_weight * (v_t_cond - v_t_uncond)
        else:
            suffix_inputs = {"x_t": noise, "timestep": t_tensor, "target_time": s_tensor, "prefix_pad_masks": prefix_outputs["prefix_pad_masks"]}
            for name in prefix_outputs.keys():
                if name != "prefix_pad_masks":
                    suffix_inputs[name] = prefix_outputs[name]
            v_t = self.suffix_runner.run(suffix_inputs)["v_t"]
            
        # Clamp velocity magnitude matching self.config.snapflow_clamp
        snapflow_clamp = getattr(self.config, "snapflow_clamp", 20.0)
        v_t = v_t.clamp(-snapflow_clamp, snapflow_clamp)
        
        # Euler step: x_0 = noise + (-1.0) * v_t
        actions = noise - v_t
        
        if "action" in self.config.output_features:
            action_dim = self.config.output_features["action"].shape[0]
            actions = actions[:, :, :action_dim]
            
        return actions

class RECAPTRTPolicy:
    def __init__(self, engine_dir: str, original_policy):
        self.config = original_policy.config
        self.prefix_runner = TRTEngineRunner(os.path.join(engine_dir, "smolvla_prefix.trt"))
        self.suffix_runner = TRTEngineRunner(os.path.join(engine_dir, "smolvla_suffix.trt"))
        self.num_steps = self.config.num_steps
        self.use_adv = getattr(self.config, "use_advantage_conditioning", False)
        self.adv_pos_id = getattr(original_policy.model, "adv_pos_id", None)
        self.cfg_weight = getattr(self.config, "cfg_weight", 1.0)

    def predict_action_chunk(self, observation: dict, noise: torch.Tensor = None) -> torch.Tensor:
        device = torch.device("cuda")
        image_keys = sorted([k for k in observation.keys() if k.startswith("observation.images")])
        
        # 1. Run prefix engine
        prefix_inputs = {}
        for idx, k in enumerate(image_keys):
            prefix_inputs[self.prefix_runner.engine.get_tensor_name(idx)] = observation[k]
        
        state = observation["observation.state"]
        if state.ndim > 2:
            state = state[:, -1, :]
        prefix_inputs["state"] = pad_or_truncate(state, self.config.max_state_dim, 0.0)
        
        # Pad language inputs to static shape of 64
        lang_tokens = observation["observation.language.tokens"]
        lang_masks = observation["observation.language.attention_mask"]
        
        pad_token_id = getattr(self.config, "pad_token_id", 1)
        if pad_token_id is None:
            pad_token_id = 1
            
        lang_tokens = pad_or_truncate(lang_tokens, 64, pad_token_id)
        lang_masks = pad_or_truncate(lang_masks, 64, False)
        
        bsize = observation["observation.state"].shape[0]
        use_cfg = self.use_adv and self.cfg_weight != 1.0 and self.cfg_weight != 0.0
        
        if use_cfg:
            # Run unconditional prefix pass
            adv_tokens_uncond = torch.zeros((bsize, 1), dtype=torch.long, device=device)
            lang_tokens_uncond = torch.cat([lang_tokens, adv_tokens_uncond], dim=1)
            lang_masks_uncond = torch.cat([lang_masks, torch.zeros_like(adv_tokens_uncond, dtype=torch.bool)], dim=1)
            
            prefix_inputs_uncond = prefix_inputs.copy()
            prefix_inputs_uncond["lang_tokens"] = lang_tokens_uncond
            prefix_inputs_uncond["lang_masks"] = lang_masks_uncond
            prefix_outputs_uncond = self.prefix_runner.run(prefix_inputs_uncond)
            
            # Run conditional prefix pass
            adv_tokens_cond = torch.full((bsize, 1), self.adv_pos_id, dtype=torch.long, device=device)
            lang_tokens_cond = torch.cat([lang_tokens, adv_tokens_cond], dim=1)
            lang_masks_cond = torch.cat([lang_masks, torch.ones_like(adv_tokens_cond, dtype=torch.bool)], dim=1)
            
            prefix_inputs_cond = prefix_inputs.copy()
            prefix_inputs_cond["lang_tokens"] = lang_tokens_cond
            prefix_inputs_cond["lang_masks"] = lang_masks_cond
            prefix_outputs_cond = self.prefix_runner.run(prefix_inputs_cond)
        else:
            if self.use_adv and self.adv_pos_id is not None:
                adv_tokens = torch.full((bsize, 1), self.adv_pos_id, dtype=torch.long, device=device)
                lang_tokens = torch.cat([lang_tokens, adv_tokens], dim=1)
                lang_masks = torch.cat([lang_masks, torch.ones_like(adv_tokens, dtype=torch.bool)], dim=1)
            
            prefix_inputs["lang_tokens"] = lang_tokens
            prefix_inputs["lang_masks"] = lang_masks
            prefix_outputs = self.prefix_runner.run(prefix_inputs)
            
        # 2. Denoising loop
        if noise is None:
            noise = torch.randn(bsize, self.config.chunk_size, self.config.max_action_dim, device=device)
            
        dt = -1.0 / self.num_steps
        x_t = noise
        
        for step in range(self.num_steps):
            t = 1.0 + step * dt
            t_tensor = torch.tensor([t], dtype=torch.float32, device=device).expand(bsize)
            
            if use_cfg:
                # Suffix uncond
                suffix_inputs_uncond = {"x_t": x_t, "timestep": t_tensor, "prefix_pad_masks": prefix_outputs_uncond["prefix_pad_masks"]}
                for name in prefix_outputs_uncond.keys():
                    if name != "prefix_pad_masks":
                        suffix_inputs_uncond[name] = prefix_outputs_uncond[name]
                v_t_uncond = self.suffix_runner.run(suffix_inputs_uncond)["v_t"]
                
                # Suffix cond
                suffix_inputs_cond = {"x_t": x_t, "timestep": t_tensor, "prefix_pad_masks": prefix_outputs_cond["prefix_pad_masks"]}
                for name in prefix_outputs_cond.keys():
                    if name != "prefix_pad_masks":
                        suffix_inputs_cond[name] = prefix_outputs_cond[name]
                v_t_cond = self.suffix_runner.run(suffix_inputs_cond)["v_t"]
                
                # CFG Blend: uncond + w * (cond - uncond)
                v_t = v_t_uncond + self.cfg_weight * (v_t_cond - v_t_uncond)
            else:
                suffix_inputs = {"x_t": x_t, "timestep": t_tensor, "prefix_pad_masks": prefix_outputs["prefix_pad_masks"]}
                for name in prefix_outputs.keys():
                    if name != "prefix_pad_masks":
                        suffix_inputs[name] = prefix_outputs[name]
                v_t = self.suffix_runner.run(suffix_inputs)["v_t"]
                
            x_t = x_t + dt * v_t
            
        actions = x_t
        if "action" in self.config.output_features:
            action_dim = self.config.output_features["action"].shape[0]
            actions = actions[:, :, :action_dim]
            
        return actions

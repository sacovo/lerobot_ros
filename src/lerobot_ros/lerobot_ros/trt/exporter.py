import os
import torch
import torch.nn as nn
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

class PolicyONNXWrapper(nn.Module):
    def __init__(self, policy, obs_keys):
        super().__init__()
        self.policy = policy
        self.obs_keys = obs_keys

    def forward(self, *obs_tensors):
        obs = {k: v for k, v in zip(self.obs_keys, obs_tensors)}
        return self.policy.predict_action_chunk(obs)

def export_act(policy, sample_observation, output_path):
    """
    Exports an ACT policy to ONNX format.
    """
    obs_keys = sorted(list(sample_observation.keys()))
    wrapper = PolicyONNXWrapper(policy, obs_keys)
    
    device = next(policy.parameters()).device
    wrapper.eval().to(device)

    dummy_tensors = tuple(sample_observation[k] for k in obs_keys)
    
    print(f"Exporting ACT ONNX to {output_path}...")
    print(f"Input keys: {obs_keys}")
    
    torch.onnx.export(
        wrapper,
        dummy_tensors,
        output_path,
        input_names=obs_keys,
        output_names=["action_chunk"],
        opset_version=18,
    )
    print("Export complete.")

class SmolVLAPrefixWrapper(nn.Module):
    def __init__(self, policy, image_keys):
        super().__init__()
        self.policy = policy
        self.model = policy.model
        self.image_keys = image_keys
        self.num_images = len(image_keys)

    def forward(self, *args):
        # args: num_images tensors, then state, lang_tokens, lang_masks
        images = list(args[:self.num_images])
        state = args[self.num_images]
        lang_tokens = args[self.num_images + 1]
        lang_masks = args[self.num_images + 2]

        # Build batch dict for prepare methods
        batch = {}
        for idx, k in enumerate(self.image_keys):
            batch[k] = images[idx]
        batch["observation.state"] = state

        if hasattr(self.policy, "prepare_images"):
            prepared_images, img_masks = self.policy.prepare_images(batch)
        else:
            from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
            prepared_images, img_masks = SmolVLAPolicy.prepare_images(self.policy, batch)
        prepared_state = state[:, -1, :] if state.ndim > 2 else state

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            prepared_images, img_masks, lang_tokens, lang_masks, state=prepared_state
        )
        att_2d = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        pos_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, cache = self.model.vlm_with_expert.forward(
            attention_mask=att_2d,
            position_ids=pos_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        flat = []
        for i in range(len(cache)):
            flat.extend([cache[i]['key_states'], cache[i]['value_states']])
        return tuple(flat) + (prefix_pad_masks,)

class SmolVLASuffixWrapper(nn.Module):
    def __init__(self, policy, num_layers):
        super().__init__()
        self.model = policy.model
        self.num_layers = num_layers

    def forward(self, x_t, timestep, prefix_pad_masks, *kv_flat):
        cache = {}
        for i in range(self.num_layers):
            cache[i] = {
                'key_states': kv_flat[2*i],
                'value_states': kv_flat[2*i+1]
            }

        v_t = self.model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=cache,
            x_t=x_t,
            timestep=timestep,
        )
        return v_t

class RECAPSnapflowSuffixWrapper(nn.Module):
    def __init__(self, policy, num_layers):
        super().__init__()
        self.model = policy.model
        self.num_layers = num_layers

    def forward(self, x_t, timestep, target_time, prefix_pad_masks, *kv_flat):
        cache = {}
        for i in range(self.num_layers):
            cache[i] = {
                'key_states': kv_flat[2*i],
                'value_states': kv_flat[2*i+1]
            }

        v_t = self.model.denoise_step_snapflow(
            x_t=x_t,
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=cache,
            timestep=timestep,
            target_time=target_time,
        )
        return v_t

def clean_create_sinusoidal_pos_embedding(time, dimension, min_period, max_period, device="cpu"):
    """Computes float32 sine-cosine positional embedding without None indexing/float64 to keep ONNX clean."""
    import math
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")
    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = torch.float32
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    scaling_factor = 1.0 / period * 2 * math.pi
    scaling_factor = scaling_factor.unsqueeze(0)
    time = time.unsqueeze(1)
    
    sin_input = scaling_factor * time
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb

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

def export_smolvla(policy, sample_observation, output_dir):
    """
    Exports a SmolVLA or SmolVLARECAP policy to prefix and suffix ONNX models.
    """
    device = next(policy.parameters()).device
    policy.eval().to(device)

    # 1. Patch sinusoidal pos embedding to use float32 and avoid None indexing (fixes unsqueeze double weights error)
    import lerobot.policies.smolvla.modeling_smolvla as m_smol
    m_smol.create_sinusoidal_pos_embedding = clean_create_sinusoidal_pos_embedding
    try:
        import lerobot_policy_smolvla_rl.modeling_smolvla_recap as m_recap
        m_recap.create_sinusoidal_pos_embedding = clean_create_sinusoidal_pos_embedding
    except ImportError:
        pass

    # 2. Patch vision model embeddings to avoid in-place indexing (Where op type mismatch)
    vlm = policy.model.vlm_with_expert.vlm
    embeddings_module = None
    if hasattr(vlm, "model") and hasattr(vlm.model, "vision_model"):
        embeddings_module = vlm.model.vision_model.embeddings
    elif hasattr(vlm, "vision_model"):
        embeddings_module = vlm.vision_model.embeddings
        
    if embeddings_module is not None:
        print("Patching SmolVLMVisionEmbeddings.forward to avoid in-place indexing...")
        
        def clean_embeddings_forward(pixel_values, patch_attention_mask):
            batch_size, _, max_im_h, max_im_w = pixel_values.shape

            patch_embeds = embeddings_module.patch_embedding(pixel_values)
            embeddings = patch_embeds.flatten(2).transpose(1, 2)

            boundaries = torch.arange(
                1 / embeddings_module.num_patches_per_side, 1.0, 1 / embeddings_module.num_patches_per_side, device=pixel_values.device
            )

            nb_patches_h = patch_attention_mask[:, :, 0].sum(dim=1)  # (batch_size,)
            nb_patches_w = patch_attention_mask[:, 0, :].sum(dim=1)  # (batch_size,)

            step_h = 1.0 / nb_patches_h  # (batch_size,)
            step_w = 1.0 / nb_patches_w  # (batch_size,)

            max_patches_h = patch_attention_mask.size(1)
            max_patches_w = patch_attention_mask.size(2)
            h_indices = torch.arange(max_patches_h, device=pixel_values.device, dtype=torch.float32)
            w_indices = torch.arange(max_patches_w, device=pixel_values.device, dtype=torch.float32)

            fractional_coords_h = h_indices[None, :] * step_h[:, None]
            fractional_coords_w = w_indices[None, :] * step_w[:, None]

            fractional_coords_h = torch.clamp(fractional_coords_h, max=(1.0 - 1e-6))
            fractional_coords_w = torch.clamp(fractional_coords_w, max=(1.0 - 1e-6))

            fractional_coords_h = fractional_coords_h.to(pixel_values.dtype)
            fractional_coords_w = fractional_coords_w.to(pixel_values.dtype)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = bucket_coords_h[:, :, None] * embeddings_module.num_patches_per_side + bucket_coords_w[:, None, :]
            pos_ids = pos_ids.reshape(batch_size, -1)

            # Avoid in-place assignment by using torch.where
            mask_flat = patch_attention_mask.view(batch_size, -1)
            position_ids = torch.where(mask_flat, pos_ids.to(torch.long), torch.zeros_like(pos_ids, dtype=torch.long))

            embeddings = embeddings + embeddings_module.position_embedding(position_ids)
            return embeddings
            
        embeddings_module.forward = clean_embeddings_forward

    # 3. Patch apply_rope to avoid in-place slice assignments which cause TRT parser errors
    import lerobot.policies.smolvla.smolvlm_with_expert as m_expert
    
    def clean_apply_rope(x, positions, max_wavelength=10000.0):
        d_half = x.shape[-1] // 2
        device = x.device
        dtype = x.dtype
        x = x.to(torch.float32)

        freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
        timescale = max_wavelength**freq_exponents
        radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

        radians = radians[..., None, :]

        sin = torch.sin(radians)
        cos = torch.cos(radians)

        x1, x2 = x.split(d_half, dim=-1)
        res1 = x1 * cos - x2 * sin
        res2 = x2 * cos + x1 * sin
        res = torch.cat([res1, res2], dim=-1)

        return res.to(dtype)
        
    m_expert.apply_rope = clean_apply_rope

    # 4. Patch pad_tensor inside lerobot to bypass index_put when max_len is 0
    original_pad_tensor = m_smol.pad_tensor
    
    def clean_pad_tensor(tensor, max_len, pad_value=0):
        if max_len <= 0:
            return tensor
        return original_pad_tensor(tensor, max_len, pad_value)
        
    m_smol.pad_tensor = clean_pad_tensor

    # Bypass CFG blending inside the traced model
    policy.model._in_cfg_mode = False
    
    # 6. Export prefix pass
    image_keys = sorted([k for k in sample_observation.keys() if k.startswith("observation.images")])

    prefix_wrapper = SmolVLAPrefixWrapper(policy, image_keys).to(device).eval()
    
    images = [sample_observation[k] for k in image_keys]
    state = sample_observation["observation.state"]
    if state.ndim > 2:
        state = state[:, -1, :]
    state = pad_or_truncate(state, policy.config.max_state_dim, 0.0)
    lang_tokens = sample_observation["observation.language.tokens"]
    lang_masks = sample_observation["observation.language.attention_mask"]
    
    # Pad language inputs to static shape (1, 64) before export
    pad_token_id = getattr(policy.model.vlm_with_expert.vlm.config, "pad_token_id", 1)
    if pad_token_id is None:
        pad_token_id = 1
    lang_tokens = pad_or_truncate(lang_tokens, 64, pad_token_id)
    lang_masks = pad_or_truncate(lang_masks, 64, False)
    
    # Append advantage conditioning token if active on python side
    if getattr(policy.config, "use_advantage_conditioning", False):
        adv_pos_id = getattr(policy.model, "adv_pos_id", None)
        if adv_pos_id is not None:
            adv_tokens = torch.full((lang_tokens.shape[0], 1), adv_pos_id, dtype=torch.long, device=lang_tokens.device)
            lang_tokens = torch.cat([lang_tokens, adv_tokens], dim=1)
            lang_masks = torch.cat([lang_masks, torch.ones_like(adv_tokens, dtype=torch.bool)], dim=1)
            
    prefix_dummy = tuple(images) + (state, lang_tokens, lang_masks)
    
    prefix_onnx_path = os.path.join(output_dir, "smolvla_prefix.onnx")
    print(f"Exporting prefix ONNX to {prefix_onnx_path}...")
    
    # Find names for inputs
    input_names = image_keys + ["state", "lang_tokens", "lang_masks"]
    
    # Compute output names
    # Run once to get cache size
    with torch.no_grad():
        prefix_out = prefix_wrapper(*prefix_dummy)
    
    num_cache_tensors = len(prefix_out) - 1
    num_layers = num_cache_tensors // 2
    
    output_names = []
    for i in range(num_layers):
        output_names.extend([f"key_{i}", f"val_{i}"])
    output_names.append("prefix_pad_masks")
    
    torch.onnx.export(
        prefix_wrapper,
        prefix_dummy,
        prefix_onnx_path,
        input_names=input_names,
        output_names=output_names,
        opset_version=18,
    )
    print("Prefix ONNX export complete.")
    
    # 7. Export suffix pass
    snapflow = getattr(policy.config, "snapflow_enabled", False)
    
    noise = policy.model.sample_noise((1, policy.config.chunk_size, policy.config.max_action_dim), device)
    timestep = torch.tensor([1.0], device=device)
    
    kv_flat = prefix_out[:-1]
    prefix_pad_masks = prefix_out[-1]
    
    suffix_onnx_path = os.path.join(output_dir, "smolvla_suffix.onnx")
    print(f"Exporting suffix ONNX to {suffix_onnx_path} (snapflow={snapflow})...")
    
    if snapflow:
        suffix_wrapper = RECAPSnapflowSuffixWrapper(policy, num_layers).to(device).eval()
        target_time = torch.tensor([0.0], device=device)
        suffix_dummy = (noise, timestep, target_time, prefix_pad_masks) + kv_flat
        
        suffix_input_names = ["x_t", "timestep", "target_time", "prefix_pad_masks"] + output_names[:-1]
    else:
        suffix_wrapper = SmolVLASuffixWrapper(policy, num_layers).to(device).eval()
        suffix_dummy = (noise, timestep, prefix_pad_masks) + kv_flat
        
        suffix_input_names = ["x_t", "timestep", "prefix_pad_masks"] + output_names[:-1]
        
    torch.onnx.export(
        suffix_wrapper,
        suffix_dummy,
        suffix_onnx_path,
        input_names=suffix_input_names,
        output_names=["v_t"],
        opset_version=18,
    )
    print("Suffix ONNX export complete.")

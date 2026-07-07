import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors

def validate_policy_accuracy(policy_pt, policy_trt, dataset_repo_id, dataset_root=None, num_samples=50):
    """
    Validates accuracy of policy_trt against policy_pt reference over num_samples from the dataset.
    Prints stats and returns max_abs_error.
    """
    print(f"Loading dataset '{dataset_repo_id}' for validation...")
    dataset = LeRobotDataset(dataset_repo_id, root=dataset_root)
    
    # Select sample indices distributed across the dataset
    indices = [i * (len(dataset) // num_samples) for i in range(num_samples)]
    print(f"Evaluating {num_samples} samples...")
    
    # We must import recap first to make sure its processors are registered if needed
    try:
        import lerobot_policy_smolvla_rl.modeling_smolvla_recap
    except ImportError:
        pass
        
    preprocessor, _ = make_pre_post_processors(
        policy_pt.config,
        pretrained_path=policy_pt.config.pretrained_path,
        preprocessor_overrides={"device_processor": {"device": "cuda"}}
    )
    
    pt_actions = []
    trt_actions = []
    
    policy_pt.eval().to("cuda")
    
    abs_errors = []
    rel_errors = []
    
    for idx in indices:
        sample = dataset[idx]
        batch = {k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v for k, v in sample.items()}
        batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        
        with torch.no_grad():
            obs = preprocessor(batch)
            
            # Check if it's a flow-matching model that needs a fixed noise tensor
            if hasattr(policy_pt, "config") and hasattr(policy_pt.config, "max_action_dim"):
                torch.manual_seed(idx)
                noise = torch.randn(
                    (1, policy_pt.config.chunk_size, policy_pt.config.max_action_dim),
                    dtype=torch.float32,
                    device="cuda"
                )
                action_pt = policy_pt.predict_action_chunk(obs, noise=noise)
                action_trt = policy_trt.predict_action_chunk(obs, noise=noise)
            else:
                action_pt = policy_pt.predict_action_chunk(obs)
                action_trt = policy_trt.predict_action_chunk(obs)
            
            pt_actions.append(action_pt.cpu())
            trt_actions.append(action_trt.cpu())
            
            abs_err = (action_pt.cpu() - action_trt.cpu()).abs()
            abs_errors.append(abs_err)
            
            rel_err = abs_err / (action_pt.cpu().abs() + 1e-8)
            rel_errors.append(rel_err)

    all_abs_err = torch.cat(abs_errors)
    all_rel_err = torch.cat(rel_errors)
    
    max_abs = all_abs_err.max().item()
    mean_abs = all_abs_err.mean().item()
    max_rel = all_rel_err.max().item()
    
    all_pt = torch.cat(pt_actions)
    all_trt = torch.cat(trt_actions)
    
    print("\n--- Accuracy Validation Report ---")
    print(f"Max absolute error:  {max_abs:.6f}")
    print(f"Mean absolute error: {mean_abs:.6f}")
    print(f"Max relative error:  {max_rel:.4f}")
    print(f"PyTorch action mean: {all_pt.mean(0).tolist()}")
    print(f"TRT action mean:     {all_trt.mean(0).tolist()}")
    print(f"PyTorch action std:  {all_pt.std(0).tolist()}")
    print(f"TRT action std:      {all_trt.std(0).tolist()}")
    print("---------------------------------\n")
    
    return max_abs, mean_abs, max_rel

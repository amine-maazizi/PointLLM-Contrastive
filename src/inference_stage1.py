"""
Inference script for Stage 1 trained models.

This script loads a trained projector and runs inference on point clouds
using the Point-BERT + Projector + LLM pipeline.
"""

import torch
import numpy as np
import os
import sys
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoModelForCausalLM, AutoTokenizer
from point_bert import load_point_bert
from checkpoint_utils import load_projector_checkpoint, load_special_token_embeddings
from train_stage1 import extract_point_features_sequence
from load_data import load_point_clouds


def preprocess_pointcloud(pointcloud: np.ndarray, num_points: int = 1024) -> torch.Tensor:
    """
    Preprocess point cloud for Point-BERT.
    
    Args:
        pointcloud: Raw point cloud (N, 3)
        num_points: Number of points to sample
    
    Returns:
        Preprocessed tensor (1, num_points, 3)
    """
    points = np.array(pointcloud)
    
    # Take only xyz if more columns exist
    if points.shape[1] > 3:
        points = points[:, :3]
    
    # Resample to fixed number of points
    if points.shape[0] >= num_points:
        indices = np.random.choice(points.shape[0], num_points, replace=False)
    else:
        indices = np.random.choice(points.shape[0], num_points, replace=True)
    points = points[indices]
    
    # Normalize to unit sphere
    centroid = np.mean(points, axis=0)
    points = points - centroid
    max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    if max_dist > 0:
        points = points / max_dist
    
    return torch.from_numpy(points).float().unsqueeze(0)


def run_inference(
    model_name: str,
    checkpoint_path: str,
    pointcloud: np.ndarray,
    point_encoder,
    device: str = "cuda",
    prompt: str = "Describe this 3D object:",
    max_new_tokens: int = 100,
    temperature: float = 0.7,
    hf_token: Optional[str] = None
) -> str:
    """
    Run inference with trained Stage 1 model.
    
    Args:
        model_name: HuggingFace model name
        checkpoint_path: Path to trained projector checkpoint
        pointcloud: Input point cloud (N, 3)
        point_encoder: Loaded Point-BERT encoder
        device: Device to run on
        prompt: Text prompt for the LLM
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        hf_token: HuggingFace API token
    
    Returns:
        Generated text description
    """
    print(f"\nLoading LLM: {model_name}...")
    
    # Load LLM and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Add special tokens
    special_tokens = {"additional_special_tokens": ["<p_start>", "<p_end>"]}
    tokenizer.add_special_tokens(special_tokens)
    llm.resize_token_embeddings(len(tokenizer))
    
    print(f"Loading projector from: {checkpoint_path}...")
    
    # Load projector
    projector, metadata = load_projector_checkpoint(checkpoint_path, device=device)
    projector.eval()
    
    # Load special token embeddings
    embedding_layer = llm.get_input_embeddings()
    special_ids = load_special_token_embeddings(checkpoint_path, embedding_layer, device)
    
    p_start_id = special_ids['p_start_id']
    p_end_id = special_ids['p_end_id']
    
    print(f"\nProjector loaded:")
    print(f"  Input: (B, {projector.num_tokens}, {projector.input_dim})")
    print(f"  Output: (B, {projector.num_tokens}, {projector.output_dim})")
    print(f"  Params: {projector.get_num_params():,}")
    
    if metadata.get('epoch'):
        print(f"\nCheckpoint info:")
        print(f"  Epoch: {metadata['epoch']}")
        print(f"  Best val loss: {min(metadata['val_losses']):.4f}" if metadata['val_losses'] else "  N/A")
    
    # Preprocess point cloud
    print(f"\nPreprocessing point cloud...")
    points = preprocess_pointcloud(pointcloud).to(device)
    print(f"Point cloud shape: {points.shape}")
    
    # Extract point features (frozen encoder)
    print(f"Extracting point features...")
    with torch.no_grad():
        point_features = extract_point_features_sequence(point_encoder, points)
        print(f"Point features shape: {point_features.shape}")
    
    # Project to LLM space (trained projector)
    print(f"Projecting to LLM space...")
    with torch.no_grad():
        point_tokens = projector(point_features.float())  # (B, num_tokens, llm_dim)
        point_tokens = point_tokens.half()  # Convert to fp16 for LLM
        print(f"Point tokens shape: {point_tokens.shape}")
    
    # Get special token embeddings
    p_start_embed = embedding_layer.weight[p_start_id].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
    p_end_embed = embedding_layer.weight[p_end_id].unsqueeze(0).unsqueeze(0)  # (1, 1, D)
    
    # Tokenize prompt
    prompt_tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    prompt_embeds = embedding_layer(prompt_tokens)  # (1, L, D)
    
    # Concatenate: <p_start> + point_tokens + <p_end> + prompt
    inputs_embeds = torch.cat([
        p_start_embed,      # (1, 1, D)
        point_tokens,       # (1, num_tokens, D)
        p_end_embed,        # (1, 1, D)
        prompt_embeds       # (1, L, D)
    ], dim=1)
    
    print(f"Input embeddings shape: {inputs_embeds.shape}")
    print(f"\nGenerating response...")
    
    # Generate
    with torch.no_grad():
        outputs = llm.generate(
            inputs_embeds=inputs_embeds,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    # Skip the input tokens (we only want the generated part)
    generated_tokens = outputs[0][inputs_embeds.shape[1]:]
    text_out = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return text_out


def main():
    parser = argparse.ArgumentParser(description="Run inference with Stage 1 trained model")
    parser.add_argument("--model", type=str, required=True, 
                       help="Model name (e.g., TinyLlama-1.1B-Chat-v1.0)")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to projector checkpoint")
    parser.add_argument("--point-bert", type=str, 
                       default=os.path.join(os.path.dirname(__file__), "..", "models", "Point-BERT.pth"),
                       help="Path to Point-BERT checkpoint")
    parser.add_argument("--dvae", type=str, 
                       default=os.path.join(os.path.dirname(__file__), "..", "models", "dVAE.pth"),
                       help="Path to dVAE checkpoint")
    parser.add_argument("--data-dir", type=str, 
                       default=os.path.join(os.path.dirname(__file__), "..", "data"),
                       help="Directory with point cloud data")
    parser.add_argument("--sample-idx", type=int, default=5,
                       help="Index of point cloud sample to use")
    parser.add_argument("--prompt", type=str, default="Describe this 3D object:",
                       help="Prompt for the LLM")
    parser.add_argument("--max-tokens", type=int, default=100,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda/cpu)")
    
    args = parser.parse_args()
    
    # Determine full model name
    if "/" not in args.model:
        # Map short names to full names
        model_map = {
            "TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "pythia-1b": "EleutherAI/pythia-1b",
            "gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B",
        }
        model_name = model_map.get(args.model, args.model)
    else:
        model_name = args.model
    
    print("="*60)
    print("STAGE 1 INFERENCE")
    print("="*60)
    print(f"Model: {model_name}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load HF token
    token_path = os.path.join(os.path.dirname(__file__), '..', '.hf_token')
    HF_TOKEN = None
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            HF_TOKEN = f.read().strip()
    
    # Load Point-BERT encoder
    print("\nLoading Point-BERT encoder...")
    point_encoder, _ = load_point_bert(
        checkpoint_path=args.point_bert,
        dvae_path=args.dvae,
        device=args.device
    )
    print("Point-BERT loaded!")
    
    # Load point cloud
    print(f"\nLoading point clouds from {args.data_dir}...")
    point_clouds, labels = load_point_clouds()
    
    if args.sample_idx >= len(point_clouds):
        print(f"Error: Sample index {args.sample_idx} out of range (max: {len(point_clouds)-1})")
        return
    
    pointcloud = point_clouds[args.sample_idx]
    true_label = labels[args.sample_idx]
    
    print(f"Selected sample {args.sample_idx}")
    print(f"True label: {true_label}")
    print(f"Point cloud shape: {pointcloud.shape}")
    
    # Run inference
    output = run_inference(
        model_name=model_name,
        checkpoint_path=args.checkpoint,
        pointcloud=pointcloud,
        point_encoder=point_encoder,
        device=args.device,
        prompt=args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        hf_token=HF_TOKEN
    )
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"True label: {true_label}")
    print(f"Prompt: {args.prompt}")
    print(f"\nGenerated response:")
    print(output)
    print("="*60)


if __name__ == "__main__":
    # python .\src\inference_stage1.py --model TinyLlama-1.1B-Chat-v1.0 --checkpoint .\checkpoints\stage1\TinyLlama-1.1B-Chat-v1.0\best_TinyLlama-1.1B-Chat-v1.0_projector.pth --sample-idx 5
    main()

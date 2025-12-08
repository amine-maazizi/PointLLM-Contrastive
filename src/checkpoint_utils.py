"""
Utilities for loading and managing Stage 1 checkpoints.
"""

import torch
import os
import json
from typing import Dict, Optional, Tuple
from projector.projector_sequence import SequenceProjectionLayer


def save_initial_projector(
    projector: SequenceProjectionLayer,
    save_dir: str,
    seed: int = 42
) -> str:
    """
    Save initial projector weights before training.
    This is used as the starting point for all LLMs.
    
    Args:
        projector: Initialized projector with random weights
        save_dir: Directory to save initial weights
        seed: Random seed used for initialization
    
    Returns:
        Path to saved checkpoint
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'projector_state_dict': projector.state_dict(),
        'projector_config': {
            'input_dim': projector.input_dim,
            'hidden_dim': projector.hidden_dim,
            'output_dim': projector.output_dim,
            'num_tokens': projector.num_tokens
        },
        'seed': seed,
        'note': 'Initial projector weights before Stage 1 training'
    }
    
    path = os.path.join(save_dir, "initial_projector_weights.pth")
    torch.save(checkpoint, path)
    print(f"Saved initial projector weights to: {path}")
    
    return path


def load_projector_checkpoint(
    checkpoint_path: str,
    device: str = "cuda"
) -> Tuple[SequenceProjectionLayer, Dict]:
    """
    Load projector from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
    
    Returns:
        projector: Loaded projector model
        metadata: Dictionary with checkpoint metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract config
    config = checkpoint['projector_config']
    
    # Create projector
    projector = SequenceProjectionLayer(
        input_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim'],
        num_tokens=config['num_tokens']
    ).to(device)
    
    # Load weights
    projector.load_state_dict(checkpoint['projector_state_dict'])
    
    # Extract metadata
    metadata = {
        'epoch': checkpoint.get('epoch', None),
        'global_step': checkpoint.get('global_step', None),
        'model_name': checkpoint.get('model_name', None),
        'seed': checkpoint.get('seed', None),
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
    }
    
    # Load special token embeddings if present
    if 'special_token_embeddings' in checkpoint:
        metadata['special_token_embeddings'] = checkpoint['special_token_embeddings']
        metadata['special_token_ids'] = checkpoint['special_token_ids']
    
    return projector, metadata


def load_special_token_embeddings(
    checkpoint_path: str,
    llm_embedding_layer,
    device: str = "cuda"
) -> Dict[str, int]:
    """
    Load and apply special token embeddings from checkpoint to LLM.
    
    Args:
        checkpoint_path: Path to checkpoint with special token embeddings
        llm_embedding_layer: LLM's embedding layer
        device: Device
    
    Returns:
        Dictionary with special token IDs
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if 'special_token_embeddings' not in checkpoint:
        raise ValueError("Checkpoint does not contain special token embeddings")
    
    # Get embeddings and IDs
    special_embeds = checkpoint['special_token_embeddings']
    special_ids = checkpoint['special_token_ids']
    
    # Apply to LLM
    with torch.no_grad():
        llm_embedding_layer.weight[special_ids['p_start_id']] = special_embeds['p_start'].to(device)
        llm_embedding_layer.weight[special_ids['p_end_id']] = special_embeds['p_end'].to(device)
    
    print(f"Loaded special token embeddings for <p_start> (ID: {special_ids['p_start_id']}) and <p_end> (ID: {special_ids['p_end_id']})")
    
    return special_ids


def save_training_summary(
    save_dir: str,
    model_results: Dict[str, Dict]
):
    """
    Save summary of training across all models.
    
    Args:
        save_dir: Directory to save summary
        model_results: Dictionary mapping model names to their training results
    """
    summary = {
        'num_models': len(model_results),
        'models': {}
    }
    
    for model_name, results in model_results.items():
        model_short = model_name.split('/')[-1]
        summary['models'][model_short] = {
            'full_name': model_name,
            'num_epochs': len(results.get('train_losses', [])),
            'final_train_loss': results.get('train_losses', [])[-1] if results.get('train_losses') else None,
            'final_val_loss': results.get('val_losses', [])[-1] if results.get('val_losses') else None,
            'best_val_loss': min(results.get('val_losses', [])) if results.get('val_losses') else None,
            'checkpoint_path': results.get('best_checkpoint_path', None)
        }
    
    path = os.path.join(save_dir, "training_summary.json")
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSaved training summary to: {path}")
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    for model_short, info in summary['models'].items():
        print(f"\n{model_short}:")
        print(f"  Epochs: {info['num_epochs']}")
        print(f"  Best Val Loss: {info['best_val_loss']:.4f}" if info['best_val_loss'] else "  Best Val Loss: N/A")
        print(f"  Checkpoint: {info['checkpoint_path']}")
    print("="*60)


def get_checkpoint_info(checkpoint_path: str) -> Dict:
    """
    Get information about a checkpoint without loading the model.
    
    Args:
        checkpoint_path: Path to checkpoint
    
    Returns:
        Dictionary with checkpoint information
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    info = {
        'path': checkpoint_path,
        'model_name': checkpoint.get('model_name', 'Unknown'),
        'epoch': checkpoint.get('epoch', None),
        'global_step': checkpoint.get('global_step', None),
        'projector_config': checkpoint.get('projector_config', {}),
        'has_special_tokens': 'special_token_embeddings' in checkpoint,
        'train_losses': checkpoint.get('train_losses', []),
        'val_losses': checkpoint.get('val_losses', []),
    }
    
    if info['train_losses']:
        info['final_train_loss'] = info['train_losses'][-1]
    if info['val_losses']:
        info['final_val_loss'] = info['val_losses'][-1]
        info['best_val_loss'] = min(info['val_losses'])
    
    return info


def print_checkpoint_info(checkpoint_path: str):
    """Print detailed information about a checkpoint."""
    info = get_checkpoint_info(checkpoint_path)
    
    print("\n" + "="*60)
    print("CHECKPOINT INFORMATION")
    print("="*60)
    print(f"Path: {info['path']}")
    print(f"Model: {info['model_name']}")
    print(f"Epoch: {info['epoch']}")
    print(f"Global Step: {info['global_step']}")
    print(f"\nProjector Config:")
    for key, value in info['projector_config'].items():
        print(f"  {key}: {value}")
    print(f"\nHas Special Tokens: {info['has_special_tokens']}")
    
    if info.get('final_train_loss'):
        print(f"\nFinal Train Loss: {info['final_train_loss']:.4f}")
    if info.get('final_val_loss'):
        print(f"Final Val Loss: {info['final_val_loss']:.4f}")
    if info.get('best_val_loss'):
        print(f"Best Val Loss: {info['best_val_loss']:.4f}")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    # Example: Check a checkpoint
    import sys
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        if os.path.exists(checkpoint_path):
            print_checkpoint_info(checkpoint_path)
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
    else:
        print("Usage: python checkpoint_utils.py <checkpoint_path>")

"""
T-SNE Visualization Script for Trained Point-LLM Models

This script visualizes the effect of Stage 1 training on point cloud alignment:
1. INITIAL PROJECTOR: Point features projected with randomly initialized weights
2. TRAINED PROJECTOR: Point features projected with trained checkpoint weights
3. TEXT EMBEDDINGS: LLM text token embeddings for comparison

The visualization shows whether training moves point features closer to the LLM's text space.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cdist
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from load_data import load_point_clouds
from point_bert import load_point_bert
from inference_stage1 import preprocess_pointcloud
from train_stage1 import extract_point_features_sequence
from projector.projector_sequence import SequenceProjectionLayer
from checkpoint_utils import load_projector_checkpoint
import glob
import argparse


def load_hf_token():
    """Load HuggingFace token from file."""
    token_path = os.path.join(os.path.dirname(__file__), '..', '.hf_token')
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            return f.read().strip()
    return None


def find_latest_checkpoint(model_dir: str) -> tuple:
    """Find the latest epoch checkpoint and initial weights for a model."""
    # Find all epoch checkpoints
    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint_*_epoch*.pth"))
    best_checkpoint = os.path.join(model_dir, f"best_{os.path.basename(model_dir)}_projector.pth")
    initial_weights = os.path.join(model_dir, "initial_projector_weights.pth")
    
    if not checkpoints and not os.path.exists(best_checkpoint):
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    
    if not os.path.exists(initial_weights):
        raise FileNotFoundError(f"Initial weights not found: {initial_weights}")
    
    # Use best checkpoint if available, otherwise latest epoch
    if os.path.exists(best_checkpoint):
        latest = best_checkpoint
    else:
        # Extract epoch numbers and find max
        epochs = [int(f.split("epoch")[-1].replace(".pth", "")) for f in checkpoints]
        latest_epoch = max(epochs)
        latest = [f for f in checkpoints if f"epoch{latest_epoch}.pth" in f][0]
    
    return initial_weights, latest


def get_available_models(checkpoint_dir: str = "./checkpoints/stage1") -> dict:
    """Get list of available trained models."""
    models = {}
    if not os.path.exists(checkpoint_dir):
        return models
    
    for item in os.listdir(checkpoint_dir):
        model_path = os.path.join(checkpoint_dir, item)
        if os.path.isdir(model_path):
            try:
                initial, trained = find_latest_checkpoint(model_path)
                # Map short name to full HF name
                full_name_map = {
                    "TinyLlama-1.1B-Chat-v1.0": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                    "pythia-1b": "EleutherAI/pythia-1b",
                    "gpt-neo-1.3B": "EleutherAI/gpt-neo-1.3B",
                }
                full_name = full_name_map.get(item, item)
                models[item] = {
                    'full_name': full_name,
                    'initial': initial,
                    'trained': trained,
                    'dir': model_path
                }
            except FileNotFoundError:
                continue
    
    return models


def get_text_embeddings(llm, tokenizer, texts: list, device: str):
    """
    Get text token embeddings from the LLM.
    
    Args:
        llm: Loaded LLM model
        tokenizer: LLM tokenizer
        texts: List of text strings
        device: Device to run on
    
    Returns:
        embeddings: Token embeddings (total_tokens, emb_dim)
        token_labels: List of token strings
    """
    all_embeddings = []
    token_labels = []
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            embeddings = llm.get_input_embeddings()(inputs.input_ids)
        
        # Get individual tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        all_embeddings.append(embeddings[0].cpu().float().numpy())
        token_labels.extend(tokens)
    
    return np.vstack(all_embeddings), token_labels


def project_point_features(point_features: torch.Tensor, projector, device: str) -> np.ndarray:
    """
    Project point features using a projector.
    
    Args:
        point_features: Point features (B, num_tokens, feat_dim)
        projector: Projection layer
        device: Device to run on
    
    Returns:
        projected: Projected features averaged over tokens (B, output_dim)
    """
    with torch.no_grad():
        # Project all tokens
        projected = projector(point_features.to(device).float())  # (B, num_tokens, output_dim)
        # Average over tokens to get single vector per point cloud
        projected_avg = projected.mean(dim=1)  # (B, output_dim)
    
    return projected_avg.cpu().numpy()


def compute_alignment_metrics(point_features: np.ndarray, text_features: np.ndarray):
    """
    Compute alignment metrics between point and text feature clusters.
    
    Args:
        point_features: Point cloud features (N, D)
        text_features: Text features (M, D)
    
    Returns:
        dict: Dictionary containing alignment metrics
    """
    # Centroid distance
    point_centroid = np.mean(point_features, axis=0)
    text_centroid = np.mean(text_features, axis=0)
    centroid_distance = np.linalg.norm(point_centroid - text_centroid)
    
    # Average pairwise distance between clusters
    pairwise_distances = cdist(point_features, text_features, metric='euclidean')
    avg_pairwise_distance = np.mean(pairwise_distances)
    min_pairwise_distance = np.min(pairwise_distances)
    
    # Cosine similarity
    cos_sim = cosine_similarity(point_features, text_features)
    avg_cosine_similarity = np.mean(cos_sim)
    max_cosine_similarity = np.max(cos_sim)
    
    # Intra-cluster variance
    point_variance = np.mean(np.var(point_features, axis=0))
    text_variance = np.mean(np.var(text_features, axis=0))
    
    return {
        'centroid_distance': centroid_distance,
        'avg_pairwise_distance': avg_pairwise_distance,
        'min_pairwise_distance': min_pairwise_distance,
        'avg_cosine_similarity': avg_cosine_similarity,
        'max_cosine_similarity': max_cosine_similarity,
        'point_variance': point_variance,
        'text_variance': text_variance
    }


def visualize_tsne_comparison(
    point_features_initial: np.ndarray,
    point_features_generative: np.ndarray,
    point_features_contrastive: np.ndarray,
    text_embeddings: np.ndarray,
    point_labels: list = None,
    save_path: str = None,
    perplexity: int = 30,
    random_state: int = 42
):
    """
    Compare initial (untrained) vs generative vs contrastive training in a single plot.
    
    Args:
        point_features_initial: Point features with initial/untrained projector (N, D)
        point_features_generative: Point features with generative trained projector (N, D)
        point_features_contrastive: Point features with contrastive trained projector (N, D)
        text_embeddings: Text token embeddings (M, D)
        point_labels: Labels for point clouds
        save_path: Path to save the figure
        perplexity: t-SNE perplexity parameter
        random_state: Random seed for reproducibility
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 11))
    print("\n" + "=" * 50)
    print("COMPARING INITIAL vs GENERATIVE vs CONTRASTIVE")
    print("=" * 50)
    
    # Combine all features for joint t-SNE
    n_points = point_features_generative.shape[0]
    combined = np.vstack([
        point_features_initial,
        point_features_generative,
        point_features_contrastive,
        text_embeddings
    ])
    print(f"Combined features shape: {combined.shape}")
    print(f"  Initial (untrained) points: {point_features_initial.shape[0]}")
    print(f"  Generative points: {point_features_generative.shape[0]}")
    print(f"  Contrastive points: {point_features_contrastive.shape[0]}")
    print(f"  Text embeddings: {text_embeddings.shape[0]}")
    
    # Compute alignment metrics
    init_metrics = compute_alignment_metrics(point_features_initial, text_embeddings)
    gen_metrics = compute_alignment_metrics(point_features_generative, text_embeddings)
    cont_metrics = compute_alignment_metrics(point_features_contrastive, text_embeddings)
    
    print(f"\nInitial (untrained) alignment metrics:")
    print(f"  Centroid distance: {init_metrics['centroid_distance']:.4f}")
    print(f"  Avg pairwise distance: {init_metrics['avg_pairwise_distance']:.4f}")
    print(f"  Avg cosine similarity: {init_metrics['avg_cosine_similarity']:.4f}")
    
    print(f"\nGenerative alignment metrics:")
    print(f"  Centroid distance: {gen_metrics['centroid_distance']:.4f}")
    print(f"  Avg pairwise distance: {gen_metrics['avg_pairwise_distance']:.4f}")
    print(f"  Avg cosine similarity: {gen_metrics['avg_cosine_similarity']:.4f}")
    
    print(f"\nContrastive alignment metrics:")
    print(f"  Centroid distance: {cont_metrics['centroid_distance']:.4f}")
    print(f"  Avg pairwise distance: {cont_metrics['avg_pairwise_distance']:.4f}")
    print(f"  Avg cosine similarity: {cont_metrics['avg_cosine_similarity']:.4f}")
    
    # Run t-SNE on combined features
    print("\nRunning t-SNE on combined features...")
    perp = min(perplexity, combined.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perp, random_state=random_state, n_iter=1000)
    combined_2d = tsne.fit_transform(combined)
    
    # Split back into separate groups
    init_2d = combined_2d[:n_points]
    gen_2d = combined_2d[n_points:2*n_points]
    cont_2d = combined_2d[2*n_points:3*n_points]
    text_2d = combined_2d[3*n_points:]
    
    # Plot all four groups
    ax.scatter(init_2d[:, 0], init_2d[:, 1], 
               c='#95a5a6', s=100, alpha=0.6, label='Initial (Untrained)', 
               marker='o', edgecolors='black', linewidths=0.5)
    ax.scatter(gen_2d[:, 0], gen_2d[:, 1], 
               c='#3498db', s=120, alpha=0.7, label='Generative Loss', 
               marker='o', edgecolors='black', linewidths=0.5)
    ax.scatter(cont_2d[:, 0], cont_2d[:, 1], 
               c='#9b59b6', s=120, alpha=0.7, label='Contrastive Loss', 
               marker='s', edgecolors='black', linewidths=0.5)
    ax.scatter(text_2d[:, 0], text_2d[:, 1], 
               c='#e74c3c', s=80, alpha=0.6, label='Text Embeddings', 
               marker='^', edgecolors='black', linewidths=0.3)
    
    ax.set_title('TinyLlama: Untrained vs Generative vs Contrastive Training\nt-SNE Visualization of Feature Alignment', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add metrics comparison text box
    metrics_text = (
        f"Initial (Untrained):\n"
        f"  Centroid dist: {init_metrics['centroid_distance']:.2f}\n"
        f"  Avg cos sim: {init_metrics['avg_cosine_similarity']:.3f}\n\n"
        f"Generative Loss:\n"
        f"  Centroid dist: {gen_metrics['centroid_distance']:.2f}\n"
        f"  Avg cos sim: {gen_metrics['avg_cosine_similarity']:.3f}\n\n"
        f"Contrastive Loss:\n"
        f"  Centroid dist: {cont_metrics['centroid_distance']:.2f}\n"
        f"  Avg cos sim: {cont_metrics['avg_cosine_similarity']:.3f}"
    )
    ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Print comparison
    print("\n" + "=" * 50)
    print("THREE-WAY COMPARISON")
    print("=" * 50)
    print(f"{'Metric':<25} {'Initial':>12} {'Generative':>12} {'Contrastive':>12} {'Best':>12}")
    print("-" * 80)
    
    for key in ['centroid_distance', 'avg_pairwise_distance', 'avg_cosine_similarity']:
        init_val = init_metrics[key]
        gen_val = gen_metrics[key]
        cont_val = cont_metrics[key]
        
        if key == 'avg_cosine_similarity':
            best_val = max(init_val, gen_val, cont_val)
            if best_val == cont_val:
                best = "Contrastive"
            elif best_val == gen_val:
                best = "Generative"
            else:
                best = "Initial"
        else:
            best_val = min(init_val, gen_val, cont_val)
            if best_val == cont_val:
                best = "Contrastive"
            elif best_val == gen_val:
                best = "Generative"
            else:
                best = "Initial"
        
        print(f"{key:<25} {init_val:>12.4f} {gen_val:>12.4f} {cont_val:>12.4f} {best:>12}")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to {save_path}")
    
    plt.show()
    
    return init_metrics, gen_metrics, cont_metrics


def main():
    parser = argparse.ArgumentParser(description="Visualize Stage 1 training effect with t-SNE")
    parser.add_argument("--checkpoint-dir", type=str, 
                       default=os.path.join(os.path.dirname(__file__), "..", "checkpoints", "stage1"),
                       help="Directory containing trained model checkpoints")
    parser.add_argument("--point-bert", type=str,
                       default=os.path.join(os.path.dirname(__file__), "..", "models", "Point-BERT.pth"),
                       help="Path to Point-BERT checkpoint")
    parser.add_argument("--dvae", type=str,
                       default=os.path.join(os.path.dirname(__file__), "..", "models", "dVAE.pth"),
                       help="Path to dVAE checkpoint")
    parser.add_argument("--num-samples", type=int, default=50,
                       help="Number of point cloud samples to visualize")
    parser.add_argument("--perplexity", type=int, default=10,
                       help="t-SNE perplexity parameter")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to run on (cuda/cpu)")
    
    args = parser.parse_args()
    
    DEVICE = args.device if torch.cuda.is_available() else "cpu"
    HF_TOKEN = load_hf_token()
    
    print("=" * 70)
    print("T-SNE VISUALIZATION: GENERATIVE VS CONTRASTIVE")
    print("Comparing TinyLlama training with different loss functions")
    print("=" * 70)
    
    # Fixed model paths for TinyLlama
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    GENERATIVE_DIR = os.path.join(args.checkpoint_dir, "TinyLlama-1.1B-Chat-v1.0")
    CONTRASTIVE_DIR = os.path.join(args.checkpoint_dir, "TinyLlama-1.1B-Chat-v1.0_contrastive")
    
    # Check if both exist
    if not os.path.exists(GENERATIVE_DIR):
        print(f"ERROR: Generative checkpoint not found: {GENERATIVE_DIR}")
        return
    if not os.path.exists(CONTRASTIVE_DIR):
        print(f"ERROR: Contrastive checkpoint not found: {CONTRASTIVE_DIR}")
        return
    
    # Find checkpoints
    INITIAL_WEIGHTS, GENERATIVE_WEIGHTS = find_latest_checkpoint(GENERATIVE_DIR)
    _, CONTRASTIVE_WEIGHTS = find_latest_checkpoint(CONTRASTIVE_DIR)
    
    print(f"\n{'='*70}")
    print(f"Model: {MODEL_NAME}")
    print(f"Initial (untrained) weights: {os.path.basename(INITIAL_WEIGHTS)}")
    print(f"Generative checkpoint: {os.path.basename(GENERATIVE_WEIGHTS)}")
    print(f"Contrastive checkpoint: {os.path.basename(CONTRASTIVE_WEIGHTS)}")
    print(f"{'='*70}")
    
    # Load point clouds
    print("\n[2/7] Loading point clouds...")
    point_clouds, labels = load_point_clouds()
    
    # Use first N point clouds for visualization
    num_samples = min(args.num_samples, len(point_clouds))
    selected_pcs = point_clouds[:num_samples]
    selected_labels = labels[:num_samples]
    
    print(f"    Using {num_samples} point clouds: {', '.join(selected_labels[:5])}{'...' if len(selected_labels) > 5 else ''}")
    
    # Generate training texts from labels (instruction + target format used in training)
    SAMPLE_TEXTS = []
    for label in selected_labels:
        # Format: "Describe this 3D object: {label}"
        # instruction = "Describe this 3D object:"
        target = label
        full_text = target # f"{instruction} {target}"
        SAMPLE_TEXTS.append(full_text)
    
    print(f"    Generated {len(SAMPLE_TEXTS)} text samples from labels for embedding")
    
    # Load Point-BERT encoder
    print("\n[3/7] Loading Point-BERT encoder...")
    point_encoder, _ = load_point_bert(
        checkpoint_path=args.point_bert,
        dvae_path=args.dvae,
        device=DEVICE
    )
    print("    Point-BERT loaded!")
    
    # Extract point features
    print("\n[4/7] Extracting Point-BERT features...")
    point_features_list = []
    
    for i, pc in enumerate(selected_pcs):
        points = preprocess_pointcloud(pc).to(DEVICE)
        features = extract_point_features_sequence(point_encoder, points)  # (1, num_tokens, 384)
        point_features_list.append(features)
        if (i + 1) % 5 == 0:
            print(f"    Processed {i+1}/{num_samples}...")
    
    point_features = torch.cat(point_features_list, dim=0)  # (N, num_tokens, 384)
    print(f"    Point features shape: {point_features.shape}")
    
    # Load LLM
    print(f"\n[5/7] Loading LLM ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    llm = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,  # Use fp32 for consistency
        device_map="auto",
        token=HF_TOKEN
    )
    
    llm_emb_dim = llm.get_input_embeddings().weight.shape[1]
    print(f"    LLM embedding dimension: {llm_emb_dim}")
    
    # Get text embeddings
    print("\n[6/7] Extracting text embeddings from LLM...")
    text_embeddings, text_labels = get_text_embeddings(llm, tokenizer, SAMPLE_TEXTS, DEVICE)
    print(f"    Text embeddings shape: {text_embeddings.shape}")
    print(f"    Text samples: {SAMPLE_TEXTS[:3]}{'...' if len(SAMPLE_TEXTS) > 3 else ''}")
    
    # Load projectors and project features
    print("\n[7/7] Loading projectors and projecting features...")
    
    # Initial (untrained) projector
    print("    Loading initial (untrained) projector...")
    init_projector, _ = load_projector_checkpoint(INITIAL_WEIGHTS, device=DEVICE)
    init_projector.eval()
    init_projected = project_point_features(point_features, init_projector, DEVICE)
    print(f"    Initial projection shape: {init_projected.shape}")
    
    # Generative projector
    print("    Loading generative projector...")
    gen_projector, gen_metadata = load_projector_checkpoint(GENERATIVE_WEIGHTS, device=DEVICE)
    gen_projector.eval()
    gen_projected = project_point_features(point_features, gen_projector, DEVICE)
    print(f"    Generative projection shape: {gen_projected.shape}")
    if gen_metadata.get('epoch'):
        print(f"    Generative checkpoint: Epoch {gen_metadata['epoch']}, Val loss: {min(gen_metadata.get('val_losses', [999])):.4f}")
    
    # Contrastive projector
    print("    Loading contrastive projector...")
    cont_projector, cont_metadata = load_projector_checkpoint(CONTRASTIVE_WEIGHTS, device=DEVICE)
    cont_projector.eval()
    cont_projected = project_point_features(point_features, cont_projector, DEVICE)
    print(f"    Contrastive projection shape: {cont_projected.shape}")
    if cont_metadata.get('epoch'):
        print(f"    Contrastive checkpoint: Epoch {cont_metadata['epoch']}, Val loss: {min(cont_metadata.get('val_losses', [999])):.4f}")
    
    # Visualize with t-SNE
    print("\n" + "=" * 70)
    print("Running t-SNE visualization...")
    print("=" * 70)
    
    save_path = "./tsne_comparison_all.png"
    
    init_metrics, gen_metrics, cont_metrics = visualize_tsne_comparison(
        point_features_initial=init_projected,
        point_features_generative=gen_projected,
        point_features_contrastive=cont_projected,
        text_embeddings=text_embeddings,
        point_labels=selected_labels,
        save_path=save_path,
        perplexity=args.perplexity
    )
    
    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("✓ Visualization complete!")
    print(f"\nFigure saved to: {save_path}")
    print(f"\nComparing {num_samples} point clouds projected with:")
    print("  - Initial (Untrained projector)")
    print("  - Generative Loss (language modeling)")
    print("  - Contrastive Loss (InfoNCE)")
    print("\nKey findings:")
    
    # Determine best approach
    findings = []
    
    # Centroid distance (lower is better)
    best_centroid = min(
        (init_metrics['centroid_distance'], 'Initial'),
        (gen_metrics['centroid_distance'], 'Generative'),
        (cont_metrics['centroid_distance'], 'Contrastive')
    )
    findings.append(f"  ✓ {best_centroid[1]} has CLOSEST centroid to text space ({best_centroid[0]:.2f})")
    
    # Cosine similarity (higher is better)
    best_cosine = max(
        (init_metrics['avg_cosine_similarity'], 'Initial'),
        (gen_metrics['avg_cosine_similarity'], 'Generative'),
        (cont_metrics['avg_cosine_similarity'], 'Contrastive')
    )
    findings.append(f"  ✓ {best_cosine[1]} has HIGHEST cosine similarity ({best_cosine[0]:.3f})")
    
    for finding in findings:
        print(finding)
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

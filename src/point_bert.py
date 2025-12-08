import torch
import numpy as np
import sys
import os
import glob

# Add Point-BERT to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Point-BERT'))

from models.Point_BERT import PointTransformer
from models.dvae import DiscreteVAE


class DVAEConfig:
    """Simple config class for DiscreteVAE initialization."""
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not installed. GLB loading will not be available.")
    print("Install with: pip install trimesh pyglet")


def load_glb_as_point_cloud(glb_path: str, num_samples: int = 10000) -> np.ndarray:
    """
    Load a GLB/GLTF file and extract point cloud by sampling from mesh surface.
    
    Args:
        glb_path: Path to the GLB file
        num_samples: Number of points to sample from the mesh surface
    
    Returns:
        Point cloud as numpy array (N, 3)
    """
    if not TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required to load GLB files. Install with: pip install trimesh pyglet")
    
    # Load the GLB file
    scene = trimesh.load(glb_path)
    
    # Handle scene vs single mesh
    if isinstance(scene, trimesh.Scene):
        # Combine all meshes in the scene
        meshes = []
        for name, geometry in scene.geometry.items():
            if isinstance(geometry, trimesh.Trimesh):
                meshes.append(geometry)
        
        if len(meshes) == 0:
            raise ValueError(f"No valid meshes found in {glb_path}")
        
        # Concatenate all meshes
        combined = trimesh.util.concatenate(meshes)
    elif isinstance(scene, trimesh.Trimesh):
        combined = scene
    else:
        raise ValueError(f"Unexpected type loaded from GLB: {type(scene)}")
    
    # Sample points from the mesh surface
    points, _ = trimesh.sample.sample_surface(combined, num_samples)
    
    return points.astype(np.float32)


def load_point_cloud(file_path: str, num_samples: int = 10000) -> np.ndarray:
    """
    Load point cloud from various file formats.
    
    Args:
        file_path: Path to the point cloud file (.npy, .txt, .glb, .gltf, .ply, .obj)
        num_samples: Number of points to sample (for mesh formats)
    
    Returns:
        Point cloud as numpy array (N, 3)
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        points = np.load(file_path)
    elif ext == '.txt':
        points = np.loadtxt(file_path, delimiter=',')
    elif ext in ['.glb', '.gltf', '.ply', '.obj', '.stl', '.off']:
        points = load_glb_as_point_cloud(file_path, num_samples)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .npy, .txt, .glb, .gltf, .ply, .obj, .stl, .off")
    
    return points.astype(np.float32)


def load_point_bert(checkpoint_path: str, dvae_path: str, num_classes: int = 40, device: str = 'cuda'):
    """
    Load Point-BERT model with pre-trained weights.
    
    Args:
        checkpoint_path: Path to Point-BERT.pth weights
        dvae_path: Path to dVAE.pth weights
        num_classes: Number of output classes (40 for ModelNet40)
        device: Device to load model on
    
    Returns:
        Loaded Point-BERT model in eval mode
    """
    # Model configuration (based on PointTransformer.yaml)
    model_config = DVAEConfig(
        trans_dim=384,
        depth=12,
        drop_path_rate=0.1,
        num_heads=6,
        encoder_dims=256,
        cls_dim=num_classes,
        group_size=32,
        num_group=64,
        loss='cdl2',
    )
    
    # dVAE configuration
    dvae_config = DVAEConfig(
        encoder_dims=256,
        num_tokens=8192,
        tokens_dims=256,  # Same as encoder_dims for dVAE
        decoder_dims=256,
        group_size=32,
        num_group=64,
    )
    
    # Load dVAE
    dvae = DiscreteVAE(dvae_config).to(device)
    
    dvae_ckpt = torch.load(dvae_path, map_location=device)
    if 'base_model' in dvae_ckpt:
        dvae_state_dict = dvae_ckpt['base_model']
    else:
        dvae_state_dict = dvae_ckpt
    
    # Handle DataParallel prefix if present in dVAE checkpoint
    new_dvae_state_dict = {}
    for k, v in dvae_state_dict.items():
        if k.startswith('module.'):
            new_dvae_state_dict[k[7:]] = v
        else:
            new_dvae_state_dict[k] = v
    
    dvae.load_state_dict(new_dvae_state_dict)
    dvae.eval()
    
    # Load Point-BERT
    model = PointTransformer(model_config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'base_model' in checkpoint:
        state_dict = checkpoint['base_model']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    
    return model, dvae


def preprocess_point_cloud(points: np.ndarray, num_points: int = 1024) -> torch.Tensor:
    """
    Preprocess point cloud for inference.
    
    Args:
        points: Input point cloud (N, 3) or (N, 6) with normals
        num_points: Number of points to sample
    
    Returns:
        Preprocessed point cloud tensor (1, num_points, 3)
    """
    # Take only xyz coordinates
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
    points = points / max_dist
    
    # Convert to tensor and add batch dimension
    points = torch.from_numpy(points).float().unsqueeze(0)
    
    return points


def run_inference(
    model,
    points: np.ndarray,
    device: str = 'cuda',
    class_names: list = None
) -> dict:
    """
    Run inference on a single point cloud.
    
    Args:
        model: Loaded Point-BERT model
        points: Input point cloud (N, 3)
        device: Device to run inference on
        class_names: Optional list of class names for output
    
    Returns:
        Dictionary with prediction results
    """
    # Preprocess
    input_tensor = preprocess_point_cloud(points).to(device)
    
    # Run inference
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=-1)
        pred_class = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_class].item()
    
    result = {
        'predicted_class': pred_class,
        'confidence': confidence,
        'probabilities': probs[0].cpu().numpy(),
    }
    
    if class_names is not None:
        result['predicted_label'] = class_names[pred_class]
    
    return result


def load_modelnet40_class_names(data_path: str = None) -> list:
    """Load ModelNet40 class names."""
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__),
            'Point-BERT/data/ModelNet/modelnet40_normal_resampled/modelnet40_shape_names.txt'
        )
    
    with open(data_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    
    return class_names


def main():
    """Example usage of Point-BERT inference."""
    import argparse
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_checkpoint = os.path.join(script_dir, '..', 'models', 'Point-BERT.pth')
    default_dvae = os.path.join(script_dir, '..', 'models', 'dVAE.pth')
    default_data_dir = os.path.join(script_dir, '..', 'data')
    
    parser = argparse.ArgumentParser(description='Point-BERT Inference')
    parser.add_argument('--input', type=str, default=None, help='Path to input point cloud (.npy, .txt, .glb, .gltf, .ply, .obj)')
    parser.add_argument('--data_dir', type=str, default=default_data_dir, help='Directory containing GLB files to process')
    parser.add_argument('--checkpoint', type=str, default=default_checkpoint, help='Path to Point-BERT weights')
    parser.add_argument('--dvae', type=str, default=default_dvae, help='Path to dVAE weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda or cpu)')
    parser.add_argument('--num_classes', type=int, default=40, help='Number of classes')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of points to sample from mesh')
    
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Load model
    print(f"Loading Point-BERT model from {args.checkpoint}...")
    model, dvae = load_point_bert(
        checkpoint_path=args.checkpoint,
        dvae_path=args.dvae,
        num_classes=args.num_classes,
        device=args.device
    )
    print("Model loaded successfully!")
    
    # Load class names
    try:
        class_names = load_modelnet40_class_names()
    except FileNotFoundError:
        class_names = None
        print("Warning: Could not load class names")
    
    # Determine input files
    if args.input:
        input_files = [args.input]
    else:
        # Load all GLB files from data directory
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), args.data_dir))
        input_files = glob.glob(os.path.join(data_dir, '*.glb'))
        input_files += glob.glob(os.path.join(data_dir, '*.gltf'))
        if not input_files:
            print(f"No GLB/GLTF files found in {data_dir}")
            return
        print(f"Found {len(input_files)} GLB/GLTF files in {data_dir}")
    
    # Process each file
    all_results = []
    for input_file in input_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(input_file)}")
        print('='*60)
        
        try:
            # Load point cloud
            points = load_point_cloud(input_file, num_samples=args.num_samples)
            print(f"Point cloud shape: {points.shape}")
            
            # Run inference
            result = run_inference(model, points, device=args.device, class_names=class_names)
            result['file'] = os.path.basename(input_file)
            all_results.append(result)
            
            # Print results
            print(f"Predicted class index: {result['predicted_class']}")
            if 'predicted_label' in result:
                print(f"Predicted label: {result['predicted_label']}")
            print(f"Confidence: {result['confidence']:.4f}")
            
            # Top 5 predictions
            top5_indices = np.argsort(result['probabilities'])[::-1][:5]
            print("\nTop 5 predictions:")
            for i, idx in enumerate(top5_indices):
                label = class_names[idx] if class_names else f"Class {idx}"
                print(f"  {i+1}. {label}: {result['probabilities'][idx]:.4f}")
                
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue
    
    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print('='*60)
        for res in all_results:
            label = res.get('predicted_label', f"Class {res['predicted_class']}")
            print(f"{res['file']}: {label} ({res['confidence']:.2%})")


if __name__ == '__main__':
    main()
    # Usage examples:
    # Process all GLB files in data folder:
    #   python point_bert.py
    # 
    # Process a single file:
    #   python point_bert.py --input ../data/"Coffee Maker.glb"
    #
    # With custom paths:
    #   python point_bert.py --checkpoint ../models/Point-BERT.pth --dvae ../models/dVAE.pth --data_dir ../data
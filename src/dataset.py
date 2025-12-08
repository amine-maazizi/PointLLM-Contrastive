import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Dict
import random


class PointCloudInstructionDataset(Dataset):
    """
    Dataset for point cloud + instruction pairs for Stage 1 training.
    Each sample contains:
    - Point cloud
    - Instruction text
    - Target response (filename without extension)
    """
    
    def __init__(
        self,
        data_dir: str,
        instructions: List[str] = None,
        num_points: int = 1024,
        max_samples: int = None
    ):
        """
        Args:
            data_dir: Directory containing .npz point cloud files
            instructions: List of instruction templates to randomly sample from
            num_points: Number of points to sample from each point cloud
            max_samples: Maximum number of samples to load (None = all)
        """
        self.data_dir = data_dir
        self.num_points = num_points
        
        # Default instruction templates
        if instructions is None:
            self.instructions = [
                "Describe this 3D object briefly.",
                "What is this 3D point cloud object?",
                "Identify this 3D shape.",
                "Summarize the 3D point cloud object briefly.",
                "What object does this point cloud represent?",
            ]
        else:
            self.instructions = instructions
        
        # Load all .npz files
        self.files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
        
        if max_samples is not None:
            self.files = self.files[:max_samples]
        
        print(f"Loaded {len(self.files)} point cloud files from {data_dir}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Dict[str, any]:
        """
        Returns:
            Dictionary containing:
            - 'points': Preprocessed point cloud tensor (num_points, 3)
            - 'instruction': Instruction string
            - 'target': Target response (filename without extension)
            - 'filename': Original filename
        """
        filename = self.files[idx]
        filepath = os.path.join(self.data_dir, filename)
        
        # Load point cloud
        data = np.load(filepath)
        points = data["points"]
        
        # Preprocess point cloud
        points = self._preprocess_points(points)
        
        # Get target (filename without extension)
        target = os.path.splitext(filename)[0]
        
        # Randomly sample an instruction
        instruction = random.choice(self.instructions)
        
        return {
            'points': points,
            'instruction': instruction,
            'target': target,
            'filename': filename
        }
    
    def _preprocess_points(self, points: np.ndarray) -> torch.Tensor:
        """
        Preprocess point cloud: resample, center, and normalize.
        
        Args:
            points: Raw point cloud (N, 3)
        
        Returns:
            Preprocessed tensor (num_points, 3)
        """
        # Take only xyz if more columns exist
        if points.shape[1] > 3:
            points = points[:, :3]
        
        # Resample to fixed number of points
        if points.shape[0] >= self.num_points:
            indices = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            indices = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[indices]
        
        # Normalize to unit sphere
        centroid = np.mean(points, axis=0)
        points = points - centroid
        max_dist = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
        if max_dist > 0:
            points = points / max_dist
        
        return torch.from_numpy(points).float()


def create_dataloaders(
    data_dir: str,
    val_samples: int = 100,
    batch_size: int = 1,
    num_points: int = 1024,
    num_workers: int = 0,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with fixed split.
    
    Args:
        data_dir: Directory containing .npz point cloud files
        val_samples: Number of samples to use for validation
        batch_size: Batch size for training
        num_points: Number of points per point cloud
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility
    
    Returns:
        train_loader, val_loader
    """
    # Set seed for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
    # Create full dataset
    full_dataset = PointCloudInstructionDataset(
        data_dir=data_dir,
        num_points=num_points
    )
    
    # Split into train and validation
    total_size = len(full_dataset)
    val_size = min(val_samples, total_size // 10)  # At most 10% for validation
    train_size = total_size - val_size
    
    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"Dataset split: {train_size} train, {val_size} validation")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset
    data_dir = "../data"
    
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        val_samples=100,
        batch_size=2
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Show first batch
    batch = next(iter(train_loader))
    print(f"\nSample batch:")
    print(f"Points shape: {batch['points'].shape}")
    print(f"Instructions: {batch['instruction']}")
    print(f"Targets: {batch['target']}")

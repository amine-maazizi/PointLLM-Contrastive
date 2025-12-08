import torch
import torch.nn as nn
from typing import Optional


class SequenceProjectionLayer(nn.Module):
    """
    3-layer MLP projector with GeLU activations for Stage 1 Feature Alignment.
    Maps point features from encoder to LLM token space as a sequence.
    This is the ONLY trainable component during Stage 1.
    """
    
    def __init__(
        self,
        input_dim: int = 384,  # Point-BERT feature dimension per token
        hidden_dim: int = 512,
        output_dim: int = 1024,  # LLM embedding dimension
        num_tokens: int = 64  # Number of point tokens from encoder
    ):
        """
        Args:
            input_dim: Dimension of each point feature from encoder (384 for Point-BERT)
            hidden_dim: Hidden dimension for projection MLP
            output_dim: Output dimension matching the LLM embedding size
            num_tokens: Number of point tokens (sequence length) from encoder
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_tokens = num_tokens
        
        # 3-layer MLP with GeLU
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()
        
        # Layer normalization for stability
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # Initialize weights with smaller scale for stability
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with small values to prevent exploding gradients."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use smaller std for initialization
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project point features to LLM token space.
        
        Args:
            x: Point features from encoder, shape (B, num_tokens, input_dim)
        
        Returns:
            Projected point tokens, shape (B, num_tokens, output_dim)
        """
        # x shape: (B, num_tokens, input_dim)
        B, N, D = x.shape
        
        # Apply MLP to each token independently
        # Reshape to (B*N, D) for efficient processing
        x = x.reshape(B * N, D)
        
        # Layer 1 with normalization
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.activation(x)
        
        # Layer 2 with normalization
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.activation(x)
        
        # Layer 3 (output layer)
        x = self.fc3(x)
        
        # Reshape back to (B, N, output_dim)
        x = x.reshape(B, N, self.output_dim)
        
        return x
    
    def get_num_params(self) -> int:
        """Return number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ProjectionLayer(nn.Module):
    """
    Original single-vector projection layer (backward compatibility).
    Maps concatenated CLS+pooled features to LLM space.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 5120):
        """
        Parameters
        ----------
        input_dim: int
            Embedding dimension of the point cloud encoder (768 for PointBERT)
        hidden_dim: int
            Hidden dimension for projection MLP
        output_dim: int
            Output dimension matching the LLM embedding size
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input point cloud embeddings, shape (B, input_dim)

        Returns
        -------
        torch.Tensor
            Projected embeddings, shape (B, output_dim)
        """
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # Test the sequence projector
    batch_size = 2
    num_tokens = 64
    input_dim = 384
    output_dim = 2048
    
    projector = SequenceProjectionLayer(
        input_dim=input_dim,
        hidden_dim=512,
        output_dim=output_dim,
        num_tokens=num_tokens
    )
    
    # Create dummy input
    x = torch.randn(batch_size, num_tokens, input_dim)
    
    # Forward pass
    output = projector(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {projector.get_num_params():,}")

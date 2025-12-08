import torch
import torch.nn as nn

class ProjectionLayer(nn.Module):
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

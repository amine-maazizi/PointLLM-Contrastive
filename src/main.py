import torch
import os
from load_data import load_point_clouds
from run_inference import run_pointcloud
from point_bert import load_point_bert

# Load HuggingFace token from file (keep .hf_token in .gitignore)
def load_hf_token():
    token_path = os.path.join(os.path.dirname(__file__), '..', '.hf_token')
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            return f.read().strip()
    return None

HF_TOKEN = load_hf_token()

# Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "./models/Point-BERT.pth"
DVAE_PATH = "./models/dVAE.pth"

# Available LLMs
MODELS = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "EleutherAI/pythia-1b",
    "EleutherAI/gpt-neo-1.3B",
]

def main():
    # Load point clouds
    print("Loading point clouds...")
    point_clouds, labels = load_point_clouds()
    
    # Load Point-BERT encoder
    print("Loading Point-BERT encoder...")
    point_encoder, _ = load_point_bert(
        checkpoint_path=CHECKPOINT_PATH,
        dvae_path=DVAE_PATH,
        device=DEVICE
    )
    print("Point-BERT loaded successfully!")
    
    # Select point cloud and LLM
    pc = point_clouds[5]
    print(f"The point cloud is a {labels[(5)]}")
    model_name = MODELS[2]
    
    # Run inference
    print(f"\nRunning inference with {model_name}...")
    output = run_pointcloud(
        model_name=model_name,
        pointcloud=pc,
        point_encoder=point_encoder,
        device=DEVICE,
        hf_token=HF_TOKEN
    )
    print(f"\nOutput:\n{output}")


if __name__ == "__main__":
    main()
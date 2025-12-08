import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from projector.projector import ProjectionLayer


def extract_point_bert_features(point_encoder, points: torch.Tensor) -> torch.Tensor:
    """
    Extract features from Point-BERT encoder.
    
    Args:
        point_encoder: Loaded Point-BERT model
        points: Point cloud tensor (B, N, 3)
    
    Returns:
        Features tensor (B, 768) - concatenated CLS and max-pooled features
    """
    with torch.no_grad():
        # Get intermediate features instead of classification logits
        neighborhood, center = point_encoder.group_divider(points)
        group_input_tokens = point_encoder.encoder(neighborhood)
        group_input_tokens = point_encoder.reduce_dim(group_input_tokens)
        
        cls_tokens = point_encoder.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = point_encoder.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = point_encoder.pos_embed(center)
        
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)
        pos = torch.cat((cls_pos, pos), dim=1)
        
        x = point_encoder.blocks(x, pos)
        x = point_encoder.norm(x)
        
        # Concatenate CLS token and max-pooled features
        features = torch.cat([x[:, 0], x[:, 1:].max(1)[0]], dim=-1)
    
    return features


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


def run_pointcloud(
    model_name: str,
    pointcloud: np.ndarray,
    point_encoder,
    device: str = "cuda",
    prompt: str = "Describe this 3D object:",
    hf_token: str = None
) -> str:
    """
    Run point cloud through Point-BERT, project to LLM space, and generate description.
    
    Args:
        model_name: HuggingFace model name
        pointcloud: Input point cloud (N, 3)
        point_encoder: Loaded Point-BERT model
        device: Device to run on
        prompt: Text prompt for the LLM
        hf_token: HuggingFace API token for gated models
    
    Returns:
        Generated text description
    """
    # Load LLM
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    llm = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )
    
    # Preprocess point cloud
    points = preprocess_pointcloud(pointcloud).to(device)
    
    # Extract Point-BERT features (768-dim)
    point_features = extract_point_bert_features(point_encoder, points)
    
    # Project to LLM embedding dimension
    llm_emb_dim = llm.get_input_embeddings().weight.shape[1]
    projector = ProjectionLayer(
        input_dim=point_features.shape[-1],  # 768
        hidden_dim=256,
        output_dim=llm_emb_dim
    ).to(device)
    
    with torch.no_grad():
        # Project point features to LLM space (B, llm_emb_dim)
        projected = projector(point_features.float())
        # Add sequence dimension (B, 1, llm_emb_dim)
        projected = projected.unsqueeze(1)
    
    # Tokenize prompt and get embeddings
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    prompt_embeds = llm.get_input_embeddings()(inputs.input_ids)
    
    # Concatenate: [point_embedding, prompt_embeddings]
    inputs_embeds = torch.cat([projected.half(), prompt_embeds], dim=1)
    
    # Generate
    outputs = llm.generate(
        inputs_embeds=inputs_embeds,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Decode output
    text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return text_out
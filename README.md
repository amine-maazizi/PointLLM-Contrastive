# PointLLM-Contrastive

Investigating contrastive loss versus generative loss for point cloud-language feature alignment in multimodal 3D understanding.

## Overview

This project implements and compares two training objectives for aligning 3D point cloud features with large language model embeddings:

- **Generative Loss**: Language modeling objective as described in the original PointLLM paper
- **Contrastive Loss**: InfoNCE (NT-Xent) objective for direct feature space alignment

Based on the PointLLM architecture from [Point Bind & Point-LLM: Aligning Point Cloud with Multi-modality for 3D Understanding, Generation, and Instruction Following](https://arxiv.org/pdf/2308.16911).

## Architecture

```
Point Cloud → Point-BERT Encoder → Projection Layer → LLM Embedding Space
                (frozen)              (trainable)         (frozen)
```

- **Encoder**: Point-BERT extracts geometric features from 3D point clouds
- **Projector**: 3-layer MLP with LayerNorm maps point features to LLM embedding dimension
- **LLM**: TinyLlama-1.1B provides the target embedding space

## Setup

### Prerequisites

```bash
pip install -r requirements.txt
```

### Point-BERT Setup

Follow the setup instructions from the [Point-BERT repository](https://github.com/Julie-tang00/Point-BERT) to obtain the pretrained checkpoint and place it in `models/Point-BERT.pth`.

### HuggingFace Token

Create a `.hf_token` file in the root directory with your HuggingFace access token for downloading gated models.

## Training

### Stage 1: Feature Alignment

Train the projection layer to align point cloud features with text embeddings:

```bash
python src/run_stage1_training.py
```

Configure loss type in `run_stage1_training.py`:
- `LOSS_TYPE = "generative"` - Language modeling objective
- `LOSS_TYPE = "contrastive"` - InfoNCE contrastive objective

Checkpoints are saved to `checkpoints/stage1/{model_name}_{loss_type}/`.

## Visualization

Compare feature alignment quality using t-SNE visualization:

```bash
python src/visualize_tsne.py
```

Generates a plot showing point cloud projections (initial, generative, contrastive) versus text embeddings in 2D space.

## Results

Alignment metrics (centroid distance, cosine similarity) are logged during training and can be visualized to compare the effectiveness of generative versus contrastive training objectives.

## References

- [PointLLM Paper](https://arxiv.org/pdf/2308.16911)
- [Point-BERT](https://github.com/Julie-tang00/Point-BERT)

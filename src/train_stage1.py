"""
Stage 1 Feature Alignment Training Script

This script fine-tunes the projector layer to align Point-BERT features with LLM embeddings.
Only the projector and special token embeddings are trainable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import json
import sys
from datetime import datetime
from tqdm import tqdm
from typing import Dict, Tuple, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset import create_dataloaders
from projector.projector_sequence import SequenceProjectionLayer
from point_bert import load_point_bert


def extract_point_features_sequence(point_encoder, points: torch.Tensor) -> torch.Tensor:
    """
    Extract sequence of point features from Point-BERT encoder.
    
    Args:
        point_encoder: Loaded Point-BERT model
        points: Point cloud tensor (B, N, 3)
    
    Returns:
        Features tensor (B, num_tokens, feature_dim) - sequence of point tokens
    """
    with torch.no_grad():
        # Get intermediate features from encoder
        neighborhood, center = point_encoder.group_divider(points)
        group_input_tokens = point_encoder.encoder(neighborhood)  # (B, num_group, encoder_dim)
        group_input_tokens = point_encoder.reduce_dim(group_input_tokens)  # (B, num_group, trans_dim)
        
        # Add CLS token
        cls_tokens = point_encoder.cls_token.expand(group_input_tokens.size(0), -1, -1)
        cls_pos = point_encoder.cls_pos.expand(group_input_tokens.size(0), -1, -1)
        pos = point_encoder.pos_embed(center)
        
        x = torch.cat((cls_tokens, group_input_tokens), dim=1)  # (B, 1+num_group, trans_dim)
        pos = torch.cat((cls_pos, pos), dim=1)
        
        # Pass through transformer blocks
        x = point_encoder.blocks(x, pos)
        x = point_encoder.norm(x)
        
        # Return all tokens (including CLS): (B, 1+num_group, trans_dim)
        # This gives us a sequence of point tokens
    
    return x


class Stage1Trainer:
    """Trainer for Stage 1 Feature Alignment."""
    
    def __init__(
        self,
        model_name: str,
        point_encoder,
        projector: SequenceProjectionLayer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        gradient_accumulation_steps: int = 4,
        loss_type: str = "generative",
        save_dir: str = "./checkpoints",
        hf_token: Optional[str] = None
    ):
        """
        Initialize Stage 1 trainer.
        
        Args:
            model_name: HuggingFace model name for LLM
            point_encoder: Pre-trained Point-BERT encoder (frozen)
            projector: Trainable projection layer
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device to train on
            lr: Learning rate
            weight_decay: Weight decay for optimizer
            gradient_accumulation_steps: Steps to accumulate gradients
            loss_type: Loss function type - "generative" or "contrastive"
            save_dir: Directory to save checkpoints
            hf_token: HuggingFace API token
        """
        self.model_name = model_name
        self.point_encoder = point_encoder
        self.projector = projector
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.loss_type = loss_type
        self.save_dir = save_dir
        
        # Validate loss type
        if loss_type not in ["generative", "contrastive"]:
            raise ValueError(f"Invalid loss_type: {loss_type}. Must be 'generative' or 'contrastive'")
        
        # Load LLM and tokenizer
        print(f"Loading LLM: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use fp32 for training stability
            device_map="auto",
            token=hf_token
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Add special tokens for point clouds
        special_tokens = {"additional_special_tokens": ["<p_start>", "<p_end>"]}
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        print(f"Added {num_added} special tokens: <p_start>, <p_end>")
        
        # Resize LLM embeddings to accommodate new tokens
        self.llm.resize_token_embeddings(len(self.tokenizer))
        
        # Get special token IDs
        self.p_start_id = self.tokenizer.convert_tokens_to_ids("<p_start>")
        self.p_end_id = self.tokenizer.convert_tokens_to_ids("<p_end>")
        
        # Freeze point encoder
        for param in self.point_encoder.parameters():
            param.requires_grad = False
        self.point_encoder.eval()
        
        # Freeze LLM except special token embeddings
        for param in self.llm.parameters():
            param.requires_grad = False
        
        # Make special token embeddings trainable
        embedding_layer = self.llm.get_input_embeddings()
        embedding_layer.weight.requires_grad = True
        
        # Initialize special token embeddings from EOS token
        with torch.no_grad():
            embedding_layer.weight[self.p_start_id] = embedding_layer.weight[self.tokenizer.eos_token_id].clone()
            embedding_layer.weight[self.p_end_id] = embedding_layer.weight[self.tokenizer.eos_token_id].clone()
        
        # Optimizer only for projector and special token embeddings
        trainable_params = [
            {'params': self.projector.parameters(), 'lr': lr},
            {'params': [embedding_layer.weight], 'lr': lr * 0.1}  # Lower LR for embeddings
        ]
        self.optimizer = torch.optim.AdamW(trainable_params, weight_decay=weight_decay)
        
        # Gradient clipping value
        self.max_grad_norm = 1.0
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.train_losses = []
        self.val_losses = []
        
        # Memory bank for contrastive learning (accumulate negatives across batches)
        if self.loss_type == "contrastive":
            self.memory_bank_size = 256  # Store embeddings from recent batches
            self.point_memory_bank = None
            self.text_memory_bank = None
            self.memory_ptr = 0
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Projector parameters: {self.projector.get_num_params():,}")
        print(f"Special token embeddings: {embedding_layer.weight[self.p_start_id:self.p_end_id+1].numel():,}")
        print(f"Loss type: {self.loss_type}")
    
    def compute_loss(self, batch: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Compute training loss for Stage 1.
        
        Supports two loss types:
        - generative: Language modeling loss on target tokens (default)
        - contrastive: Contrastive loss between point and text embeddings
        
        Args:
            batch: Batch from dataloader
        
        Returns:
            loss: Scalar loss tensor
            metrics: Dictionary of metrics
        """
        points = batch['points'].to(self.device)  # (B, N, 3)
        instructions = batch['instruction']
        targets = batch['target']
        
        batch_size = points.shape[0]
        
        # 1. Extract point features (frozen encoder)
        with torch.no_grad():
            point_features = extract_point_features_sequence(self.point_encoder, points)
            # point_features: (B, num_tokens, trans_dim)
        
        # 2. Project to LLM space (trainable projector)
        # Keep projector computation in fp32 for stability and gradient flow
        point_tokens = self.projector(point_features.float())  # (B, num_tokens, llm_dim) in fp32
        
        # Check for NaN/Inf in projector output
        if torch.isnan(point_tokens).any() or torch.isinf(point_tokens).any():
            print("WARNING: NaN/Inf detected in projector output")
            print(f"point_features stats: min={point_features.min():.4f}, max={point_features.max():.4f}, mean={point_features.mean():.4f}")
            print(f"point_tokens stats: min={point_tokens.min():.4f}, max={point_tokens.max():.4f}, mean={point_tokens.mean():.4f}")
        
        # Keep in fp32 for gradient flow - DO NOT convert to fp16 here
        
        # 3. Get special token embeddings (THESE MUST PRESERVE GRADIENTS)
        embedding_layer = self.llm.get_input_embeddings()
        # Access weight directly to preserve gradient flow for trainable special tokens
        p_start_embed = embedding_layer.weight[self.p_start_id:self.p_start_id+1].unsqueeze(0)  # (1, 1, D)
        p_end_embed = embedding_layer.weight[self.p_end_id:self.p_end_id+1].unsqueeze(0)  # (1, 1, D)
        
        # Convert to fp32 to match point_tokens
        p_start_embed = p_start_embed.float()
        p_end_embed = p_end_embed.float()
        
        # Expand for batch
        p_start_embed = p_start_embed.expand(batch_size, -1, -1)  # (B, 1, D)
        p_end_embed = p_end_embed.expand(batch_size, -1, -1)  # (B, 1, D)
        
        # 4. Compute loss based on loss_type
        if self.loss_type == "contrastive":
            return self._compute_contrastive_loss(
                point_tokens=point_tokens,
                instructions=instructions,
                targets=targets,
                embedding_layer=embedding_layer
            )
        else:  # generative
            return self._compute_generative_loss(
                point_tokens=point_tokens,
                p_start_embed=p_start_embed,
                p_end_embed=p_end_embed,
                instructions=instructions,
                targets=targets,
                embedding_layer=embedding_layer,
                batch_size=batch_size
            )
    
    def _compute_generative_loss(
        self,
        point_tokens: torch.Tensor,
        p_start_embed: torch.Tensor,
        p_end_embed: torch.Tensor,
        instructions: list,
        targets: list,
        embedding_layer,
        batch_size: int
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute generative (language modeling) loss.
        Loss is computed only on the response tokens, not the instruction.
        """
        total_loss = 0.0
        total_tokens = 0
        
        for i in range(batch_size):
            instruction = instructions[i]
            target = targets[i]
            
            # Create full text: instruction + target
            full_text = f"{instruction} {target}"
            
            # Tokenize instruction and full text
            instr_tokens = self.tokenizer(
                instruction,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)
            
            full_tokens = self.tokenizer(
                full_text,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)
            
            # Get embeddings for text tokens
            # Use detached embeddings since text encoder is frozen
            with torch.no_grad():
                instr_embeds = embedding_layer(instr_tokens).clone().detach()  # (1, L_instr, D)
                full_embeds = embedding_layer(full_tokens).clone().detach()  # (1, L_full, D)
            
            # Convert to fp32 to match point_tokens dtype
            instr_embeds = instr_embeds.float()
            full_embeds = full_embeds.float()
            
            # Concatenate: <p_start> + point_tokens[i] + <p_end> + instruction + target
            # All tensors are in fp32 with proper gradient flow
            inputs_embeds = torch.cat([
                p_start_embed[i:i+1],      # (1, 1, D) - trainable, has gradients
                point_tokens[i:i+1],        # (1, num_tokens, D) - trainable, has gradients
                p_end_embed[i:i+1],         # (1, 1, D) - trainable, has gradients
                full_embeds                 # (1, L_full, D) - frozen, no gradients
            ], dim=1)  # (1, 1 + num_tokens + 1 + L_full, D)
            
            # Verify gradient flow
            if not inputs_embeds.requires_grad:
                print(f"ERROR: inputs_embeds doesn't require grad at sample {i}")
                print(f"p_start_embed.requires_grad: {p_start_embed.requires_grad}")
                print(f"point_tokens.requires_grad: {point_tokens.requires_grad}")
                print(f"p_end_embed.requires_grad: {p_end_embed.requires_grad}")
            
            # Create labels: -100 for all non-target tokens
            # We want loss only on the target (response) tokens
            num_point_tokens = 1 + point_tokens.shape[1] + 1  # <p_start> + tokens + <p_end>
            num_instr_tokens = instr_tokens.shape[1]
            num_target_tokens = full_tokens.shape[1] - num_instr_tokens
            
            # Labels: -100 for point tokens and instruction, actual tokens for target
            labels = torch.full(
                (1, inputs_embeds.shape[1]),
                -100,
                dtype=torch.long,
                device=self.device
            )
            
            # Set target labels (after point tokens + instruction)
            target_start = num_point_tokens + num_instr_tokens
            labels[0, target_start:] = full_tokens[0, num_instr_tokens:]
            
            # Forward pass through LLM
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=False
            )
            
            loss = outputs.loss
            
            # Check for NaN loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss detected for sample {i}")
                print(f"inputs_embeds shape: {inputs_embeds.shape}")
                print(f"labels shape: {labels.shape}")
                print(f"num_target_tokens: {num_target_tokens}")
                # Skip this sample
                continue
            
            total_loss += loss
            total_tokens += num_target_tokens
        
        # Average loss over batch (handle case where all samples were skipped)
        if total_loss == 0.0 or batch_size == 0:
            avg_loss = torch.tensor(float('nan'))
        else:
            avg_loss = total_loss / batch_size
        
        metrics = {
            'loss': avg_loss.item() if not torch.isnan(avg_loss) else float('nan'),
            'num_tokens': total_tokens / batch_size if batch_size > 0 else 0
        }
        
        return avg_loss, metrics
    
    def _compute_contrastive_loss(
        self,
        point_tokens: torch.Tensor,
        instructions: list,
        targets: list,
        embedding_layer
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute contrastive loss between point and text embeddings.
        Uses InfoNCE (NT-Xent) loss with memory bank to handle batch_size=1.
        """
        batch_size = point_tokens.shape[0]
        
        # Average point tokens to get single embedding per sample
        point_embeds = point_tokens.mean(dim=1)  # (B, D)
        
        # Get text embeddings for targets (the labels we're aligning to)
        text_embeds_list = []
        for i in range(batch_size):
            target = targets[i]
            # Tokenize target
            target_tokens = self.tokenizer(
                target,
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids.to(self.device)
            
            # Get embeddings and average
            with torch.no_grad():
                target_embed = embedding_layer(target_tokens).clone().detach().float()  # (1, L, D)
            target_embed = target_embed.mean(dim=1)  # (1, D)
            text_embeds_list.append(target_embed)
        
        text_embeds = torch.cat(text_embeds_list, dim=0)  # (B, D)
        
        # Normalize embeddings
        point_embeds = F.normalize(point_embeds, p=2, dim=-1)
        text_embeds = F.normalize(text_embeds, p=2, dim=-1)
        
        # Initialize memory bank if needed
        emb_dim = point_embeds.shape[1]
        if self.point_memory_bank is None:
            self.point_memory_bank = torch.randn(self.memory_bank_size, emb_dim, device=self.device)
            self.point_memory_bank = F.normalize(self.point_memory_bank, p=2, dim=-1)
            self.text_memory_bank = torch.randn(self.memory_bank_size, emb_dim, device=self.device)
            self.text_memory_bank = F.normalize(self.text_memory_bank, p=2, dim=-1)
        
        # Compute similarity with current batch and memory bank
        temperature = 0.07
        
        # Positive pairs: current point with current text
        pos_sim = torch.sum(point_embeds * text_embeds, dim=-1) / temperature  # (B,)
        
        # Detach memory bank to prevent gradient tracking
        point_mem = self.point_memory_bank.detach()
        text_mem = self.text_memory_bank.detach()
        
        # Negative pairs: current point with all texts in memory bank
        neg_sim_point = torch.matmul(point_embeds, text_mem.t()) / temperature  # (B, M)
        
        # Negative pairs: current text with all points in memory bank
        neg_sim_text = torch.matmul(text_embeds, point_mem.t()) / temperature  # (B, M)
        
        # Construct logits: [positive, negatives]
        logits_point = torch.cat([pos_sim.unsqueeze(1), neg_sim_point], dim=1)  # (B, 1+M)
        logits_text = torch.cat([pos_sim.unsqueeze(1), neg_sim_text], dim=1)  # (B, 1+M)
        
        # Labels: 0 is the positive pair (first column)
        labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        
        # Compute loss
        loss_point_to_text = F.cross_entropy(logits_point, labels)
        loss_text_to_point = F.cross_entropy(logits_text, labels)
        loss = (loss_point_to_text + loss_text_to_point) / 2
        
        # Update memory bank (FIFO queue) - clone once to avoid in-place modification
        with torch.no_grad():
            # Clone memory banks once before updates to avoid gradient graph issues
            point_mem_update = self.point_memory_bank.clone()
            text_mem_update = self.text_memory_bank.clone()
            
            for i in range(batch_size):
                point_mem_update[self.memory_ptr] = point_embeds[i].detach()
                text_mem_update[self.memory_ptr] = text_embeds[i].detach()
                self.memory_ptr = (self.memory_ptr + 1) % self.memory_bank_size
            
            # Replace memory banks with updated versions
            self.point_memory_bank = point_mem_update
            self.text_memory_bank = text_mem_update
        
        # Compute accuracy (is positive pair ranked first?)
        with torch.no_grad():
            preds = logits_point.argmax(dim=1)
            accuracy = (preds == labels).float().mean().item()
        
        metrics = {
            'loss': loss.item(),
            'accuracy': accuracy,
            'num_tokens': 0  # Not applicable for contrastive
        }
        
        return loss, metrics
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.projector.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.epoch+1}")
        
        for step, batch in enumerate(progress_bar):
            # Compute loss
            loss, metrics = self.compute_loss(batch)
            
            # Normalize loss by accumulation steps
            loss = loss / self.gradient_accumulation_steps
            
            # Backward pass
            loss.backward()
            
            # Update weights every gradient_accumulation_steps
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(
                    self.projector.parameters(),
                    self.max_grad_norm
                )
                
                # Check for NaN gradients
                has_nan = False
                for param in self.projector.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan = True
                        break
                
                if has_nan:
                    print(f"WARNING: NaN/Inf gradient detected at step {step}, skipping optimizer step")
                    self.optimizer.zero_grad()
                else:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            epoch_loss += metrics['loss']
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f"{metrics['loss']:.4f}"})
        
        # Final optimizer step if remaining gradients
        if len(self.train_loader) % self.gradient_accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return epoch_loss / num_batches
    
    @torch.no_grad()
    def validate(self) -> float:
        """Validate on validation set."""
        self.projector.eval()
        val_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            loss, metrics = self.compute_loss(batch)
            val_loss += metrics['loss']
            num_batches += 1
        
        return val_loss / num_batches
    
    def train(self, num_epochs: int = 5):
        """
        Train for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"\nStarting Stage 1 training for {num_epochs} epochs...")
        print(f"Model: {self.model_name}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save checkpoint if best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(is_best=True)
            
            # Save regular checkpoint
            self.save_checkpoint(is_best=False)
        
        print(f"\nTraining complete!")
        print(f"Best validation loss: {best_val_loss:.4f}")
    
    def save_checkpoint(self, is_best: bool = False):
        """Save checkpoint with projector weights and special token embeddings."""
        # Get model name for filename
        model_short_name = self.model_name.split('/')[-1]
        
        # Get special token embeddings
        embedding_layer = self.llm.get_input_embeddings()
        special_token_embeds = {
            'p_start': embedding_layer.weight[self.p_start_id].cpu(),
            'p_end': embedding_layer.weight[self.p_end_id].cpu()
        }
        
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'projector_state_dict': self.projector.state_dict(),
            'special_token_embeddings': special_token_embeds,
            'special_token_ids': {
                'p_start_id': self.p_start_id,
                'p_end_id': self.p_end_id
            },
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'model_name': self.model_name,
            'loss_type': self.loss_type,
            'projector_config': {
                'input_dim': self.projector.input_dim,
                'hidden_dim': self.projector.hidden_dim,
                'output_dim': self.projector.output_dim,
                'num_tokens': self.projector.num_tokens
            }
        }
        
        if is_best:
            path = os.path.join(self.save_dir, f"best_{model_short_name}_{self.loss_type}_projector.pth")
        else:
            path = os.path.join(self.save_dir, f"checkpoint_{model_short_name}_{self.loss_type}_epoch{self.epoch+1}.pth")
        
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")
        
        # Also save training log
        log_path = os.path.join(self.save_dir, f"training_log_{model_short_name}_{self.loss_type}.json")
        log_data = {
            'model_name': self.model_name,
            'loss_type': self.loss_type,
            'epochs': self.epoch + 1,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': min(self.val_losses) if self.val_losses else None,
            'timestamp': datetime.now().isoformat()
        }
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)


def train_stage1(
    model_name: str,
    data_dir: str,
    checkpoint_path: str,
    dvae_path: str,
    save_dir: str,
    num_epochs: int = 5,
    batch_size: int = 1,
    lr: float = 1e-3,
    gradient_accumulation_steps: int = 4,
    val_samples: int = 100,
    loss_type: str = "generative",
    device: str = "cuda",
    hf_token: Optional[str] = None
):
    """
    Train Stage 1 projector for a single LLM.
    
    Args:
        model_name: HuggingFace model name
        data_dir: Directory with point cloud data
        checkpoint_path: Path to Point-BERT checkpoint
        dvae_path: Path to dVAE checkpoint
        save_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        gradient_accumulation_steps: Gradient accumulation steps
        val_samples: Number of validation samples
        loss_type: Loss function type - "generative" (default) or "contrastive"
        device: Device to train on
        hf_token: HuggingFace API token
    """
    # Load Point-BERT encoder
    print("Loading Point-BERT encoder...")
    point_encoder, _ = load_point_bert(
        checkpoint_path=checkpoint_path,
        dvae_path=dvae_path,
        device=device
    )
    print("Point-BERT loaded!")
    
    # Create dataloaders
    print("\nPreparing dataset...")
    train_loader, val_loader = create_dataloaders(
        data_dir=data_dir,
        val_samples=val_samples,
        batch_size=batch_size,
        num_points=1024
    )
    
    # Get LLM embedding dimension
    print(f"\nLoading tokenizer to get embedding dimension...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    llm_temp = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="cpu",
        token=hf_token
    )
    llm_emb_dim = llm_temp.get_input_embeddings().weight.shape[1]
    del llm_temp  # Free memory
    print(f"LLM embedding dimension: {llm_emb_dim}")
    
    # Create projector
    # Point-BERT has 65 tokens (1 CLS + 64 groups), trans_dim=384
    projector = SequenceProjectionLayer(
        input_dim=384,
        hidden_dim=512,
        output_dim=llm_emb_dim,
        num_tokens=65
    ).to(device)
    
    print(f"\nProjector architecture:")
    print(f"  Input: (B, 65, 384)")
    print(f"  Output: (B, 65, {llm_emb_dim})")
    print(f"  Trainable params: {projector.get_num_params():,}")
    
    # Create trainer
    trainer = Stage1Trainer(
        model_name=model_name,
        point_encoder=point_encoder,
        projector=projector,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        lr=lr,
        weight_decay=1e-4,
        gradient_accumulation_steps=gradient_accumulation_steps,
        loss_type=loss_type,
        save_dir=save_dir,
        hf_token=hf_token
    )
    
    # Train
    trainer.train(num_epochs=num_epochs)
    
    return trainer


if __name__ == "__main__":
    # Example usage
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    # Configuration
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    DATA_DIR = "../data"
    CHECKPOINT_PATH = "../models/Point-BERT.pth"
    DVAE_PATH = "../models/dVAE.pth"
    SAVE_DIR = "../checkpoints/stage1"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load HF token
    token_path = os.path.join(os.path.dirname(__file__), '..', '.hf_token')
    HF_TOKEN = None
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            HF_TOKEN = f.read().strip()
    
    # Train
    trainer = train_stage1(
        model_name=MODEL_NAME,
        data_dir=DATA_DIR,
        checkpoint_path=CHECKPOINT_PATH,
        dvae_path=DVAE_PATH,
        save_dir=SAVE_DIR,
        num_epochs=5,
        batch_size=1,
        lr=1e-3,
        gradient_accumulation_steps=4,
        val_samples=100,
        device=DEVICE,
        hf_token=HF_TOKEN
    )

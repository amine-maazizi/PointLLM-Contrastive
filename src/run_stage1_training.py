"""
Main Training Runner for Stage 1 Feature Alignment

This script orchestrates training of the projector on multiple LLMs.
For each LLM:
1. Initializes projector with same random seed
2. Trains only the projector and special token embeddings
3. Saves checkpoints and logs

The initial projector weights are saved before training for later evaluation.
"""

import torch
import os
import sys
from datetime import datetime
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from train_stage1 import train_stage1
from checkpoint_utils import save_initial_projector, save_training_summary
from projector.projector_sequence import SequenceProjectionLayer


def main():
    """Run Stage 1 training for all models."""
    
    # ==================== CONFIGURATION ====================
    
    # Models to train on
    MODELS = [
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        # "EleutherAI/pythia-1b",
        # "EleutherAI/gpt-neo-1.3B",
    ]
    
    # Data and model paths
    DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
    CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "Point-BERT.pth")
    DVAE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "dVAE.pth")
    
    # Output directory
    SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "stage1")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # Training hyperparameters
    NUM_EPOCHS = 5
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4  # Reduced from 1e-3 for stability
    GRADIENT_ACCUMULATION_STEPS = 4
    VAL_SAMPLES = 100
    SEED = 42
    LOSS_TYPE = "generative"  # Options: "generative" (default) or "contrastive"
    
    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    
    # Load HuggingFace token if available
    token_path = os.path.join(os.path.dirname(__file__), '..', '.hf_token')
    HF_TOKEN = None
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            HF_TOKEN = f.read().strip()
        print("HuggingFace token loaded")
    else:
        print("No HuggingFace token found (may be needed for some models)")
    
    # ==================== SAVE CONFIGURATION ====================
    
    config = {
        'models': MODELS,
        'data_dir': DATA_DIR,
        'checkpoint_path': CHECKPOINT_PATH,
        'dvae_path': DVAE_PATH,
        'save_dir': SAVE_DIR,
        'num_epochs': NUM_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
        'val_samples': VAL_SAMPLES,
        'seed': SEED,
        'loss_type': LOSS_TYPE,
        'device': DEVICE,
        'timestamp': datetime.now().isoformat()
    }
    
    config_path = os.path.join(SAVE_DIR, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved configuration to: {config_path}")
    
    # ==================== INITIALIZE PROJECTOR ====================
    
    print("\n" + "="*60)
    print("INITIALIZING PROJECTOR WITH FIXED SEED")
    print("="*60)
    
    # Set seed for reproducibility
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    
    # Create initial projector
    # We'll determine the output dimension from the first model
    # For now, use a placeholder (will be recreated with correct dim for each model)
    print(f"Random seed: {SEED}")
    print("This projector will be reinitialized with the same seed for each LLM")
    
    # Save initial projector state (we'll do this properly for each model)
    # This is just to document the seed and approach
    
    # ==================== TRAIN ON EACH MODEL ====================
    
    model_results = {}
    
    for model_idx, model_name in enumerate(MODELS):
        print("\n" + "="*60)
        print(f"MODEL {model_idx + 1}/{len(MODELS)}: {model_name}")
        print("="*60)
        
        # Reset seed for each model to ensure same initialization
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)
        
        # Create model-specific save directory
        model_short_name = model_name.split('/')[-1]
        # Add loss type suffix to directory name
        model_save_dir_name = f"{model_short_name}_{LOSS_TYPE}"
        model_save_dir = os.path.join(SAVE_DIR, model_save_dir_name)
        os.makedirs(model_save_dir, exist_ok=True)
        
        try:
            # Get LLM embedding dimension
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"\nDetermining embedding dimension for {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
            llm_temp = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="cpu",
                token=HF_TOKEN
            )
            llm_emb_dim = llm_temp.get_input_embeddings().weight.shape[1]
            del llm_temp  # Free memory
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            print(f"LLM embedding dimension: {llm_emb_dim}")
            
            # Create projector with correct dimensions
            projector = SequenceProjectionLayer(
                input_dim=384,
                hidden_dim=512,
                output_dim=llm_emb_dim,
                num_tokens=65
            ).to(DEVICE)
            
            # Save initial weights for this model
            initial_weights_path = save_initial_projector(
                projector=projector,
                save_dir=model_save_dir,
                seed=SEED
            )
            
            print(f"\nStarting training for {model_name}...")
            print(f"Checkpoints will be saved to: {model_save_dir}")
            
            # Train
            trainer = train_stage1(
                model_name=model_name,
                data_dir=DATA_DIR,
                checkpoint_path=CHECKPOINT_PATH,
                dvae_path=DVAE_PATH,
                save_dir=model_save_dir,
                num_epochs=NUM_EPOCHS,
                batch_size=BATCH_SIZE,
                lr=LEARNING_RATE,
                gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
                val_samples=VAL_SAMPLES,
                loss_type=LOSS_TYPE,
                device=DEVICE,
                hf_token=HF_TOKEN
            )
            
            # Store results
            model_results[model_name] = {
                'train_losses': trainer.train_losses,
                'val_losses': trainer.val_losses,
                'best_checkpoint_path': os.path.join(model_save_dir, f"best_{model_short_name}_{LOSS_TYPE}_projector.pth"),
                'initial_weights_path': initial_weights_path,
                'loss_type': LOSS_TYPE,
                'success': True
            }
            
            print(f"\n✓ Successfully trained {model_name}")
            print(f"  Best val loss: {min(trainer.val_losses):.4f}")
            
        except Exception as e:
            print(f"\n✗ Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            
            model_results[model_name] = {
                'success': False,
                'error': str(e)
            }
            
            # Continue with next model
            continue
        
        finally:
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # ==================== SAVE SUMMARY ====================
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Save overall summary
    save_training_summary(SAVE_DIR, model_results)
    
    # Print final summary
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    successful = [name for name, res in model_results.items() if res.get('success', False)]
    failed = [name for name, res in model_results.items() if not res.get('success', False)]
    
    print(f"\nSuccessful: {len(successful)}/{len(MODELS)}")
    for model_name in successful:
        model_short = model_name.split('/')[-1]
        best_loss = min(model_results[model_name]['val_losses'])
        print(f"  ✓ {model_short}: Best val loss = {best_loss:.4f}")
    
    if failed:
        print(f"\nFailed: {len(failed)}/{len(MODELS)}")
        for model_name in failed:
            model_short = model_name.split('/')[-1]
            error = model_results[model_name].get('error', 'Unknown error')
            print(f"  ✗ {model_short}: {error}")
    
    print(f"\nAll checkpoints saved to: {SAVE_DIR}")
    print("="*60)
    
    # Save final results
    results_path = os.path.join(SAVE_DIR, 'final_results.json')
    with open(results_path, 'w') as f:
        # Convert results to JSON-serializable format
        json_results = {}
        for model_name, res in model_results.items():
            json_results[model_name] = {
                k: v for k, v in res.items() 
                if k not in ['trainer']  # Skip non-serializable objects
            }
        json.dump(json_results, f, indent=2)
    print(f"\nFinal results saved to: {results_path}")


if __name__ == "__main__":
    print("="*60)
    print("STAGE 1 FEATURE ALIGNMENT TRAINING")
    print("Point-BERT + LLM Projector Fine-tuning")
    print("="*60)
    print()
    
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()

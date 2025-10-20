#!/usr/bin/env python3
"""
Training script for Phase A: GPT-2 with PhaseAttention

Supports both baseline GPT-2 and phase-coupled GPT-2 for character-level language modeling.
Includes comprehensive logging, checkpointing, and evaluation.

Usage:
    python src/train.py --config configs/baseline.yaml --device mps --epochs 10
    python src/train.py --config configs/phase_coupled.yaml --device mps --resume checkpoints/latest.pt
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

# Setup path for imports
script_dir = Path(__file__).parent.parent.resolve()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

# Import local modules
try:
    from src.model import GPT2Model
except ImportError:
    # Placeholder for development - will be replaced when model.py exists
    GPT2Model = None

try:
    from src.data import CharDataset, get_dataloaders
except ImportError:
    # Placeholder for development - will be replaced when data module exists
    CharDataset = None
    get_dataloaders = None

from src.phase_attention import PhaseAttention


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer for character-level language models with optional phase coupling.

    Features:
    - Gradient accumulation for effective large batch sizes
    - Learning rate warmup and decay
    - Automatic mixed precision (AMP) support
    - TensorBoard logging
    - Checkpoint saving and resuming
    - Validation with perplexity metrics
    - Phase coherence (R parameter) logging for phase-coupled models
    """

    def __init__(
        self,
        config: Dict,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        run_dir: Path,
    ):
        """
        Initialize trainer.

        Args:
            config: Training configuration dictionary
            model: The model to train (GPT2Model or PhaseGPT2Model)
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to train on (cpu, cuda, mps)
            run_dir: Directory for saving checkpoints and logs
        """
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.run_dir = run_dir

        # Training config - support both epoch-based and step-based training
        self.max_steps = config['training'].get('max_steps', None)
        self.epochs = config['training'].get('epochs', None)

        # If max_steps is specified, calculate approximate epochs
        if self.max_steps and not self.epochs:
            steps_per_epoch = len(train_loader) // config['training'].get('gradient_accumulation', 1)
            self.epochs = max(1, (self.max_steps + steps_per_epoch - 1) // steps_per_epoch)
        elif not self.epochs:
            raise ValueError("Either 'epochs' or 'max_steps' must be specified in config['training']")

        self.grad_accum_steps = config['training'].get('gradient_accumulation', 1)
        self.max_grad_norm = config['training'].get('grad_clip', 1.0)
        self.log_interval = config['training'].get('log_interval', 10)
        self.eval_interval = config['training'].get('eval_interval', 500)
        self.checkpoint_interval = config['training'].get('checkpoint_interval', 500)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler with warmup
        self.scheduler = self._create_scheduler()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=str(run_dir / 'tensorboard'))

        # Check if model is phase-coupled
        self.is_phase_coupled = hasattr(model, 'is_phase_coupled') and model.is_phase_coupled

        logger.info(f"Trainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Model type: {'Phase-coupled' if self.is_phase_coupled else 'Baseline'}")
        logger.info(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        logger.info(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        logger.info(f"  Gradient accumulation steps: {self.grad_accum_steps}")
        logger.info(f"  Effective batch size: {train_loader.batch_size * self.grad_accum_steps}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with weight decay for non-bias/layernorm params."""
        train_config = self.config['training']

        # Separate parameters into decay and no_decay groups
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            # Don't apply weight decay to biases, layer norms, or embeddings
            if any(nd in name for nd in ['bias', 'norm', 'embedding']):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        weight_decay = train_config.get('weight_decay', 0.1)
        param_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]

        learning_rate = train_config.get('learning_rate', 3e-4)
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        logger.info(f"Optimizer: AdamW (lr={learning_rate}, weight_decay={weight_decay})")

        return optimizer

    def _create_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler with warmup."""
        train_config = self.config['training']

        warmup_steps = train_config.get('warmup_steps', 0)
        if warmup_steps == 0:
            return None

        total_steps = self.max_steps if self.max_steps else (len(self.train_loader) * self.epochs // self.grad_accum_steps)

        # Linear warmup followed by cosine decay
        def lr_lambda(step):
            if step < warmup_steps:
                # Linear warmup
                return float(step) / float(max(1, warmup_steps))
            else:
                # Cosine decay
                progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
                return max(0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        logger.info(f"Scheduler: Warmup ({warmup_steps} steps) + Cosine decay (total: {total_steps} steps)")

        return scheduler

    def save_checkpoint(self, filename: str, is_best: bool = False):
        """
        Save training checkpoint.

        Args:
            filename: Name of checkpoint file
            is_best: If True, also save as 'best_model.pt'
        """
        checkpoint_path = self.run_dir / 'checkpoints' / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_path = self.run_dir / 'checkpoints' / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved: {best_path}")

        # Also save as 'latest.pt' for easy resuming
        latest_path = self.run_dir / 'checkpoints' / 'latest.pt'
        torch.save(checkpoint, latest_path)

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load training checkpoint and resume training.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        logger.info(f"Resumed from epoch {self.epoch}, step {self.global_step}")
        logger.info(f"Best validation loss so far: {self.best_val_loss:.4f}")

    def compute_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        return_metrics: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Compute loss for a batch.

        Args:
            batch: Tuple of (input_ids, target_ids)
            return_metrics: If True, return additional metrics (perplexity, R)

        Returns:
            loss: Cross-entropy loss
            metrics: Optional dict with perplexity and R parameter
        """
        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        # Forward pass
        logits = self.model(input_ids)

        # Compute cross-entropy loss
        # logits: [batch, seq_len, vocab_size]
        # targets: [batch, seq_len]
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            reduction='mean'
        )

        if return_metrics:
            metrics = {}

            # Perplexity
            metrics['perplexity'] = torch.exp(loss).item()

            # R parameter for phase-coupled models
            if self.is_phase_coupled:
                # Collect R values per PhaseAttention layer
                R_values = []
                R_per_layer = {}

                for block_idx, block in enumerate(self.model.blocks):
                    if hasattr(block, 'attention') and isinstance(block.attention, PhaseAttention):
                        R = block.attention.last_R_param.item()
                        R_values.append(R)
                        R_per_layer[f'coherence_R_layer{block_idx}'] = R

                if R_values:
                    metrics['coherence_R'] = sum(R_values) / len(R_values)
                    # Add per-layer metrics
                    metrics.update(R_per_layer)

            return loss, metrics

        return loss, None

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        total_perplexity = 0.0
        total_R = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc='Evaluating', leave=False)

        for batch in pbar:
            loss, metrics = self.compute_loss(batch, return_metrics=True)

            total_loss += loss.item()
            total_perplexity += metrics['perplexity']

            if 'coherence_R' in metrics:
                total_R += metrics['coherence_R']

            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ppl': f"{metrics['perplexity']:.2f}"
            })

        # Compute averages
        eval_metrics = {
            'val_loss': total_loss / num_batches,
            'val_perplexity': total_perplexity / num_batches,
        }

        if self.is_phase_coupled and total_R > 0:
            eval_metrics['val_coherence_R'] = total_R / num_batches

        self.model.train()

        return eval_metrics

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.

        Returns:
            Dictionary of training metrics
        """
        self.model.train()

        epoch_loss = 0.0
        epoch_tokens = 0

        pbar = tqdm(
            self.train_loader,
            desc=f'Epoch {self.epoch + 1}/{self.epochs}',
            dynamic_ncols=True
        )

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(pbar):
            # Compute loss
            loss, metrics = self.compute_loss(batch, return_metrics=True)

            # Scale loss for gradient accumulation
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            # Accumulate metrics
            epoch_loss += loss.item()
            epoch_tokens += batch[0].numel()

            # Gradient accumulation
            if (batch_idx + 1) % self.grad_accum_steps == 0:
                # Clip gradients
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm
                    )

                # Optimizer step
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.log_interval == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']

                    # Log to tensorboard
                    self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                    self.writer.add_scalar('train/perplexity', metrics['perplexity'], self.global_step)
                    self.writer.add_scalar('train/learning_rate', current_lr, self.global_step)

                    if 'coherence_R' in metrics:
                        self.writer.add_scalar('train/coherence_R', metrics['coherence_R'], self.global_step)

                        # Log per-layer R values
                        for key, value in metrics.items():
                            if key.startswith('coherence_R_layer'):
                                self.writer.add_scalar(f'train/{key}', value, self.global_step)

                    # Update progress bar
                    pbar_dict = {
                        'loss': f"{loss.item():.4f}",
                        'ppl': f"{metrics['perplexity']:.2f}",
                        'lr': f"{current_lr:.2e}",
                    }

                    if 'coherence_R' in metrics:
                        pbar_dict['R'] = f"{metrics['coherence_R']:.3f}"

                    pbar.set_postfix(pbar_dict)

                # Evaluation
                if self.global_step % self.eval_interval == 0:
                    eval_metrics = self.evaluate()

                    # Log validation metrics
                    for key, value in eval_metrics.items():
                        self.writer.add_scalar(f'eval/{key}', value, self.global_step)

                    logger.info(
                        f"Step {self.global_step}: "
                        f"val_loss={eval_metrics['val_loss']:.4f}, "
                        f"val_ppl={eval_metrics['val_perplexity']:.2f}"
                    )

                    # Save best model
                    if eval_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = eval_metrics['val_loss']
                        self.save_checkpoint(
                            f'step_{self.global_step}.pt',
                            is_best=True
                        )
                        logger.info(f"New best model! Loss: {self.best_val_loss:.4f}")

                # Checkpoint saving
                if self.global_step % self.checkpoint_interval == 0:
                    self.save_checkpoint(f'step_{self.global_step}.pt')

                # Check if we've reached max_steps
                if self.max_steps and self.global_step >= self.max_steps:
                    logger.info(f"Reached max_steps ({self.max_steps}), stopping training")
                    break

        # End of epoch metrics
        avg_loss = epoch_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0.0

        return {
            'train_loss': avg_loss,
            'tokens_processed': epoch_tokens,
            'early_stop': self.max_steps and self.global_step >= self.max_steps
        }

    def train(self):
        """Main training loop."""
        logger.info(f"\nStarting training for {self.epochs} epochs")
        logger.info(f"Total steps: {len(self.train_loader) * self.epochs // self.grad_accum_steps}")
        logger.info("=" * 80)

        start_time = time.time()

        try:
            for epoch in range(self.epoch, self.epochs):
                self.epoch = epoch

                # Train one epoch
                epoch_metrics = self.train_epoch()

                # Check if we should stop early (max_steps reached)
                if epoch_metrics.get('early_stop', False):
                    logger.info(f"Stopping training after {self.global_step} steps")
                    break

                # Evaluate at end of epoch
                eval_metrics = self.evaluate()

                # Log epoch summary
                logger.info(
                    f"\nEpoch {epoch + 1}/{self.epochs} complete:\n"
                    f"  Train loss: {epoch_metrics['train_loss']:.4f}\n"
                    f"  Val loss: {eval_metrics['val_loss']:.4f}\n"
                    f"  Val perplexity: {eval_metrics['val_perplexity']:.2f}"
                )

                if 'val_coherence_R' in eval_metrics:
                    logger.info(f"  Val coherence (R): {eval_metrics['val_coherence_R']:.4f}")

                # Note: Checkpoint saving handled by step-based checkpoints (every checkpoint_interval steps)
                # No need to save at end of each epoch to avoid disk space issues

                logger.info("=" * 80)

        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
            self.save_checkpoint('interrupted.pt')

        except Exception as e:
            logger.error(f"\nTraining failed with error: {e}")
            self.save_checkpoint('error.pt')
            raise

        finally:
            elapsed = time.time() - start_time
            logger.info(f"\nTraining completed in {elapsed / 3600:.2f} hours")
            logger.info(f"Best validation loss: {self.best_val_loss:.4f}")

            # Close tensorboard writer
            self.writer.close()


def load_config(config_path: str) -> Dict:
    """
    Load training configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded config from: {config_path}")

    return config


def get_device(device_str: str) -> torch.device:
    """
    Get PyTorch device with fallback logic.

    Args:
        device_str: Device string ('cuda', 'mps', 'cpu', or 'auto')

    Returns:
        PyTorch device
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device_str)

        # Validate device
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        elif device.type == 'mps' and not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available")

    logger.info(f"Using device: {device}")

    return device


def create_model(config: Dict, vocab_size: int, device: torch.device) -> nn.Module:
    """
    Create model from config.

    Args:
        config: Model configuration
        vocab_size: Size of vocabulary
        device: Device to create model on

    Returns:
        Initialized model

    Raises:
        ValueError: If model type is unknown
        ImportError: If model module not found
    """
    from src.model import GPT2Config

    model_config = config['model']

    if GPT2Model is None:
        raise ImportError(
            "Model class not found. Please create src/model.py with GPT2Model class."
        )

    # Create GPT2Config
    gpt2_config = GPT2Config(
        vocab_size=vocab_size,
        n_layers=model_config['n_layers'],
        d_model=model_config['d_model'],
        n_heads=model_config['n_heads'],
        d_ff=model_config.get('d_ff', 4 * model_config['d_model']),
        max_seq_len=model_config['max_seq_len'],
        dropout=model_config.get('dropout', 0.1),
        # Phase-specific parameters (used if phase attention is enabled)
        num_oscillators=model_config.get('num_oscillators', model_config['d_model'] // model_config['n_heads']),
        coupling_strength=model_config.get('coupling_strength', 1.0),
        natural_freq_std=model_config.get('natural_freq_std', 0.1),
        phase_iterations=model_config.get('phase_iterations', 10),
    )

    # Determine if using phase attention
    use_phase_attention = model_config.get('use_phase_attention', False)
    phase_layer_idx = model_config.get('phase_layer_idx', None) if use_phase_attention else None

    # Create unified GPT2Model
    model = GPT2Model(
        config=gpt2_config,
        use_phase_attention=use_phase_attention,
        phase_layer_idx=phase_layer_idx
    )

    model_type = 'phase-coupled' if use_phase_attention else 'baseline'
    logger.info(f"Created {model_type} model with {model.get_num_params():,} parameters")

    if use_phase_attention:
        logger.info(f"Phase layer info: {model.get_phase_layer_info()}")

    return model


def setup_run_directory(config: Dict, args: argparse.Namespace) -> Path:
    """
    Create run directory for logs and checkpoints.

    Args:
        config: Configuration dictionary
        args: Command line arguments

    Returns:
        Path to run directory
    """
    if args.run_name:
        run_name = args.run_name
    else:
        # Auto-generate run name
        model_type = config['model'].get('type', 'baseline')
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        run_name = f"{model_type}_{timestamp}"

    run_dir = Path(args.log_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save config to run directory
    config_save_path = run_dir / 'config.yaml'
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Config saved to: {config_save_path}")

    return run_dir


def main():
    """Main entry point for training script."""
    parser = argparse.ArgumentParser(
        description='Train GPT-2 with optional PhaseAttention',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML config file (e.g., configs/baseline.yaml)'
    )

    # Optional arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to train on'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs (overrides config)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size (overrides config)'
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (overrides config)'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--log-dir',
        type=str,
        default='runs',
        help='Directory for saving logs and checkpoints'
    )

    parser.add_argument(
        '--run-name',
        type=str,
        default=None,
        help='Name for this run (auto-generated if not provided)'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Load config
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return 1

    # Override config with command line arguments
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    if args.learning_rate is not None:
        config['training']['learning_rate'] = args.learning_rate

    # Setup device
    try:
        device = get_device(args.device)
    except RuntimeError as e:
        logger.error(f"Device error: {e}")
        return 1

    # Setup run directory
    run_dir = setup_run_directory(config, args)

    # Setup file logging
    file_handler = logging.FileHandler(run_dir / 'train.log')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    # Load data
    try:
        if get_dataloaders is None:
            raise ImportError(
                "Data loading functions not found. Please create src/data.py "
                "with get_dataloaders function."
            )

        # Prepare data loading config
        data_config = {
            'dataset': config['data'].get('dataset', 'shakespeare'),
            'batch_size': config['training']['batch_size'],
            'seq_len': config['model']['max_seq_len'],
            'num_workers': config['data'].get('num_workers', 0),
            'data_dir': Path('data/processed'),
            'raw_dir': Path('data/raw'),
        }

        train_loader, val_loader, vocab_size, char_to_idx, idx_to_char = get_dataloaders(
            config=data_config
        )

        logger.info(f"Data loaded:")
        logger.info(f"  Vocabulary size: {vocab_size}")
        logger.info(f"  Training batches: {len(train_loader)}")
        logger.info(f"  Validation batches: {len(val_loader)}")

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    # Create model
    try:
        model = create_model(config, vocab_size, device)
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return 1

    # Create trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        run_dir=run_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        try:
            trainer.load_checkpoint(args.resume)
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return 1

    # Train
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

    logger.info("\nTraining completed successfully!")
    logger.info(f"Results saved to: {run_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())

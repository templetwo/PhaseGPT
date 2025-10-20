#!/usr/bin/env python3
"""
Phase B: WikiText-2 Generalization Training with R Tracking

Trains GPT-2 models (baseline and KPC variants) on WikiText-2 with:
- Real-time order parameter R tracking
- Coherence regularization
- Anti-oversynchronization controls
- Comprehensive logging
"""

import argparse
import logging
import sys
import os
from pathlib import Path
import yaml
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

# Add Phase A src to path
sys.path.insert(0, '/home/ubuntu/phase_a_implementation/src')

from model import GPT2Model, GPT2Config
from coherence_utils import (
    compute_order_parameter,
    coherence_regularizer,
    CoherenceTracker,
    add_phase_noise,
    add_frequency_jitter
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_wikitext2(tokenizer, seq_len=512, batch_size=32):
    """Load WikiText-2 dataset."""
    from datasets import load_dataset
    
    logger.info('Loading WikiText-2 dataset...')
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    
    # Simple tokenization (char-level)
    def tokenize(examples):
        # Concatenate all text
        text = '\n'.join(examples['text'])
        # Char-level tokenization
        tokens = [ord(c) % 256 for c in text if c.strip()]
        return {'tokens': tokens}
    
    # Process dataset
    train_data = dataset['train'].map(tokenize, batched=True, remove_columns=['text'])
    val_data = dataset['validation'].map(tokenize, batched=True, remove_columns=['text'])
    
    # Convert to torch datasets - flatten the list of lists
    train_tokens = []
    for item in train_data['tokens']:
        if isinstance(item, list):
            train_tokens.extend(item)
        else:
            train_tokens.append(item)
    train_tokens = torch.tensor(train_tokens, dtype=torch.long)
    
    val_tokens = []
    for item in val_data['tokens']:
        if isinstance(item, list):
            val_tokens.extend(item)
        else:
            val_tokens.append(item)
    val_tokens = torch.tensor(val_tokens, dtype=torch.long)
    
    # Create sequences
    def create_sequences(tokens, seq_len):
        sequences = []
        for i in range(0, len(tokens) - seq_len, seq_len // 2):
            sequences.append(tokens[i:i+seq_len+1])
        return torch.stack(sequences)
    
    train_seqs = create_sequences(train_tokens, seq_len)
    val_seqs = create_sequences(val_tokens, seq_len)
    
    logger.info(f'Train sequences: {len(train_seqs)}, Val sequences: {len(val_seqs)}')
    
    train_loader = DataLoader(train_seqs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_seqs, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, 256  # vocab_size = 256 for char-level


def train_step(model, batch, optimizer, config, device, tracker=None, step=0):
    """Single training step with optional R tracking and coherence reg."""
    model.train()
    
    x = batch[:, :-1].to(device)
    y = batch[:, 1:].to(device)
    
    # Forward pass with return_info to get phases
    use_kpc = config['model'].get('use_phase_attention', False)
    
    if use_kpc:
        logits, info = model(x, return_info=True)
        phases = info.get('phases', None)
        R = info.get('R', None)
    else:
        logits = model(x)
        phases = None
        R = None
    
    # Cross-entropy loss
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        y.reshape(-1)
    )
    
    # Add coherence regularizer if enabled
    if use_kpc and phases is not None:
        coh_reg_config = config['model'].get('coherence_reg', {})
        if coh_reg_config.get('enabled', False):
            reg_loss = coherence_regularizer(
                phases,
                R_target=coh_reg_config.get('R_target', 0.45),
                lam=coh_reg_config.get('lambda', 0.1),
                mode=coh_reg_config.get('mode', 'ceiling')
            )
            loss = loss + reg_loss
    
    # Backward pass
    loss.backward()
    
    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        config['training'].get('grad_clip', 1.0)
    )
    
    optimizer.step()
    optimizer.zero_grad()
    
    # Track R if enabled
    R_mean = None
    if tracker and phases is not None:
        R_mean = tracker.update(phases, step)
    
    return loss.item(), grad_norm.item(), R_mean


@torch.no_grad()
def validate(model, val_loader, device, config):
    """Validation pass with perplexity computation."""
    model.eval()
    
    total_loss = 0
    total_tokens = 0
    R_values = []
    
    use_kpc = config['model'].get('use_phase_attention', False)
    
    for batch in val_loader:
        x = batch[:, :-1].to(device)
        y = batch[:, 1:].to(device)
        
        if use_kpc:
            logits, info = model(x, return_info=True)
            if 'R' in info and info['R'] is not None:
                R_values.append(info['R'].mean().item())
        else:
            logits = model(x)
        
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            reduction='sum'
        )
        
        total_loss += loss.item()
        total_tokens += y.numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    R_mean_val = np.mean(R_values) if R_values else None
    
    return avg_loss, perplexity, R_mean_val


def train(config_path):
    """Main training loop."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config.get('device', 'cuda'))
    logger.info(f'Using device: {device}')
    
    # Create run directory
    run_name = Path(config_path).stem
    run_dir = Path('~/PhaseB/runs').expanduser() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    train_loader, val_loader, vocab_size = load_wikitext2(
        None,
        seq_len=config['data']['seq_len'],
        batch_size=config['training']['batch_size']
    )
    
    # Create model
    model_config_dict = config['model'].copy()
    model_config_dict['vocab_size'] = vocab_size
    
    # Filter to GPT2Config fields
    from dataclasses import fields as dc_fields
    valid_fields = {f.name for f in dc_fields(GPT2Config)}
    filtered_config = {k: v for k, v in model_config_dict.items() if k in valid_fields}
    
    gpt_config = GPT2Config(**filtered_config)
    
    # Extract PhaseAttention settings
    use_phase_attention = model_config_dict.get('use_phase_attention', False)
    phase_layer_idx = model_config_dict.get('phase_layer_idx', None)
    
    model = GPT2Model(gpt_config, use_phase_attention=use_phase_attention, phase_layer_idx=phase_layer_idx)
    model = model.to(device)
    
    logger.info(f'Model created: {sum(p.numel() for p in model.parameters()):,} parameters')
    logger.info(f'PhaseAttention: {use_phase_attention} at layers {phase_layer_idx}')
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training'].get('weight_decay', 0.01)
    )
    
    # Cosine scheduler with warmup
    warmup_steps = config['training']['scheduler'].get('warmup_steps', 500)
    total_steps = len(train_loader) * config['training']['epochs']
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # R tracker if KPC enabled
    tracker = None
    if use_phase_attention and config['logging'].get('r_tracking', False):
        tracker = CoherenceTracker(log_interval=config['logging'].get('r_interval', 500))
    
    # TensorBoard
    writer = SummaryWriter(log_dir=str(run_dir / 'tensorboard'))
    
    # Training loop
    global_step = 0
    best_val_ppl = float('inf')
    
    for epoch in range(config['training']['epochs']):
        logger.info(f'\n=== Epoch {epoch+1}/{config["training"]["epochs"]} ===')
        
        epoch_loss = 0
        epoch_steps = 0
        
        for batch_idx, batch in enumerate(train_loader):
            loss, grad_norm, R_mean = train_step(
                model, batch, optimizer, config, device, tracker, global_step
            )
            
            epoch_loss += loss
            epoch_steps += 1
            global_step += 1
            
            scheduler.step()
            
            # Logging
            if global_step % config['logging']['log_interval'] == 0:
                writer.add_scalar('train/loss', loss, global_step)
                writer.add_scalar('train/grad_norm', grad_norm, global_step)
                writer.add_scalar('train/lr', scheduler.get_last_lr()[0], global_step)
                
                if R_mean is not None:
                    writer.add_scalar('train/R', R_mean, global_step)
                
                logger.info(f'Step {global_step}: loss={loss:.4f}, grad_norm={grad_norm:.4f}, R={R_mean:.4f if R_mean is not None else 0.0}')
            
            # Validation
            if global_step % config['logging']['eval_interval'] == 0:
                val_loss, val_ppl, R_mean_val = validate(model, val_loader, device, config)
                
                writer.add_scalar('val/loss', val_loss, global_step)
                writer.add_scalar('val/perplexity', val_ppl, global_step)
                
                if R_mean_val is not None:
                    writer.add_scalar('val/R', R_mean_val, global_step)
                
                logger.info(f'Validation: PPL={val_ppl:.4f}, R={R_mean_val:.4f if R_mean_val else 0.0}')
                
                # Save best model
                if val_ppl < best_val_ppl:
                    best_val_ppl = val_ppl
                    torch.save({
                        'epoch': epoch,
                        'global_step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'config': config,
                        'val_ppl': val_ppl,
                        'R_mean_val': R_mean_val
                    }, run_dir / 'best_model.pt')
                    logger.info(f'Saved best model (PPL={val_ppl:.4f})')
        
        # End of epoch
        avg_epoch_loss = epoch_loss / epoch_steps
        logger.info(f'Epoch {epoch+1} complete: avg_loss={avg_epoch_loss:.4f}')
        
        # R statistics
        if tracker:
            R_stats = tracker.get_stats()
            logger.info(f'R statistics: {R_stats}')
            tracker.reset()
    
    writer.close()
    logger.info('Training complete!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    args = parser.parse_args()
    
    train(args.config)

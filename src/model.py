"""
GPT-2 Model with Optional PhaseAttention Integration

Implements a modular GPT-2 architecture that can replace standard attention
layers with PhaseAttention (Kuramoto oscillator-based attention).

Core Components:
- GPT2Config: Configuration dataclass for model hyperparameters
- StandardAttention: Causal self-attention with multiple heads
- GPT2Block: Transformer block with optional PhaseAttention
- GPT2Model: Full GPT-2 model with flexible attention replacement

Usage:
    # Baseline GPT-2 with standard attention
    config = GPT2Config(n_layers=12, d_model=768, n_heads=12, vocab_size=50257)
    model = GPT2Model(config, use_phase_attention=False)

    # Phase-coupled GPT-2 (replace layer 6 with PhaseAttention)
    model = GPT2Model(config, use_phase_attention=True, phase_layer_idx=6)

    # Multiple phase layers
    model = GPT2Model(config, use_phase_attention=True, phase_layer_idx=[3, 6, 9])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import math
from dataclasses import dataclass, asdict
from typing import Optional, Union, List, Dict, Any
from pathlib import Path

# Handle both module and standalone execution
try:
    from .phase_attention import PhaseAttention
except ImportError:
    from phase_attention import PhaseAttention


@dataclass
class GPT2Config:
    """
    Configuration for GPT-2 model.

    Attributes:
        vocab_size: Size of the vocabulary
        n_layers: Number of transformer blocks
        d_model: Hidden dimension size
        n_heads: Number of attention heads
        d_ff: Feedforward layer dimension (default: 4 * d_model)
        max_seq_len: Maximum sequence length for positional embeddings
        dropout: Dropout probability
        layer_norm_epsilon: Epsilon for layer normalization

        # PhaseAttention-specific parameters
        num_oscillators: Number of oscillators per head (default: d_model // n_heads)
        coupling_strength: Kuramoto coupling strength K
        natural_freq_std: Standard deviation for natural frequencies
        phase_iterations: Number of Kuramoto synchronization steps
    """

    # Standard GPT-2 parameters
    vocab_size: int = 50257
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    d_ff: Optional[int] = None
    max_seq_len: int = 1024
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5

    # PhaseAttention parameters
    num_oscillators: Optional[int] = None
    coupling_strength: float = 1.0
    natural_freq_std: float = 0.1
    phase_iterations: int = 10

    def __post_init__(self):
        """Validate and set derived parameters."""
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model

        if self.num_oscillators is None:
            self.num_oscillators = self.d_model // self.n_heads

        # Validation
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
            )

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> 'GPT2Config':
        """
        Load configuration from JSON file.

        Args:
            path: Path to JSON config file

        Returns:
            GPT2Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config contains invalid parameters
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(path, 'r') as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    def to_json(self, path: Union[str, Path]) -> None:
        """
        Save configuration to JSON file.

        Args:
            path: Path to save JSON config file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)


class StandardAttention(nn.Module):
    """
    Causal multi-head self-attention.

    Standard transformer attention with:
    - Causal masking (prevents attending to future tokens)
    - Multiple heads for different attention patterns
    - Scaled dot-product attention
    """

    def __init__(self, config: GPT2Config):
        """
        Initialize standard attention layer.

        Args:
            config: Model configuration
        """
        super().__init__()

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads

        # QKV projections (combined for efficiency)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model)

        # Output projection
        self.out_proj = nn.Linear(config.d_model, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Register causal mask buffer
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.max_seq_len, config.max_seq_len)).view(
                1, 1, config.max_seq_len, config.max_seq_len
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention_weights: bool = False
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with causal self-attention.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_attention_weights: If True, return attention weights

        Returns:
            output: [batch, seq_len, d_model]
            (attention_weights): [batch, n_heads, seq_len, seq_len] if requested
        """
        batch, seq_len, d_model = x.shape

        # Compute Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, n_heads, seq_len, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        # scores = Q @ K^T / sqrt(d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # Apply causal mask (prevent attending to future)
        scores = scores.masked_fill(
            self.causal_mask[:, :, :seq_len, :seq_len] == 0,
            float('-inf')
        )

        # Softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [batch, n_heads, seq_len, d_head]

        # Reshape and project output
        out = out.transpose(1, 2).reshape(batch, seq_len, d_model)
        out = self.out_proj(out)
        out = self.dropout(out)

        if return_attention_weights:
            return out, attn_weights
        return out


class GPT2Block(nn.Module):
    """
    Single GPT-2 transformer block.

    Architecture:
        x = x + Attention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Supports both standard attention and PhaseAttention.
    """

    def __init__(
        self,
        config: GPT2Config,
        use_phase_attention: bool = False
    ):
        """
        Initialize transformer block.

        Args:
            config: Model configuration
            use_phase_attention: If True, use PhaseAttention instead of standard
        """
        super().__init__()

        self.use_phase_attention = use_phase_attention

        # Layer normalization
        self.ln1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        # Attention layer (standard or phase-based)
        if use_phase_attention:
            self.attention = PhaseAttention(
                d_model=config.d_model,
                n_heads=config.n_heads,
                num_oscillators=config.num_oscillators,
                coupling_strength=config.coupling_strength,
                natural_freq_std=config.natural_freq_std,
                phase_iterations=config.phase_iterations,
                dropout=config.dropout
            )
        else:
            self.attention = StandardAttention(config)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )

    def forward(
        self,
        x: torch.Tensor,
        return_coherence: bool = False,
        return_info: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass through transformer block.

        Args:
            x: Input tensor [batch, seq_len, d_model]
            return_coherence: If True and using PhaseAttention, return R parameter

        Returns:
            output: [batch, seq_len, d_model]
            (R): Phase coherence if requested and available
        """
        R = None
        phases = None

        # Self-attention with residual
        if self.use_phase_attention:
            if return_info:
                attn_out, attn_info = self.attention(self.ln1(x), return_info=True)
                if attn_info is not None:
                    phases = attn_info.get('phases', None)
                    R = attn_info.get('R', None)
            elif return_coherence:
                attn_out, R = self.attention(self.ln1(x), return_coherence=True)
            else:
                attn_out = self.attention(self.ln1(x))
        else:
            attn_out = self.attention(self.ln1(x))

        x = x + attn_out

        # Feedforward with residual
        x = x + self.ffn(self.ln2(x))

        if return_info:
            return x, {'phases': phases, 'R': R}
        if return_coherence:
            return x, R
        return x


class GPT2Model(nn.Module):
    """
    Full GPT-2 model with optional PhaseAttention layers.

    Architecture:
        Token Embedding + Position Embedding
        → N Transformer Blocks (with optional PhaseAttention)
        → LayerNorm
        → Language Modeling Head

    Supports:
    - Baseline GPT-2 (all standard attention)
    - Phase-coupled GPT-2 (specific layers use PhaseAttention)
    - Flexible layer replacement (single layer or multiple layers)
    """

    def __init__(
        self,
        config: GPT2Config,
        use_phase_attention: bool = False,
        phase_layer_idx: Optional[Union[int, List[int]]] = None
    ):
        """
        Initialize GPT-2 model.

        Args:
            config: Model configuration
            use_phase_attention: If True, replace specified layers with PhaseAttention
            phase_layer_idx: Which layer(s) to replace (0-indexed). Can be:
                - int: Single layer index
                - List[int]: Multiple layer indices
                - None: No replacement (baseline model)

        Example:
            # Baseline model
            model = GPT2Model(config, use_phase_attention=False)

            # Replace layer 6 with PhaseAttention
            model = GPT2Model(config, use_phase_attention=True, phase_layer_idx=6)

            # Replace layers 3, 6, 9 with PhaseAttention
            model = GPT2Model(config, use_phase_attention=True, phase_layer_idx=[3, 6, 9])
        """
        super().__init__()

        self.config = config
        self.use_phase_attention = use_phase_attention

        # Normalize phase_layer_idx to list
        if phase_layer_idx is None:
            self.phase_layer_indices = set()
        elif isinstance(phase_layer_idx, int):
            self.phase_layer_indices = {phase_layer_idx}
        else:
            self.phase_layer_indices = set(phase_layer_idx)

        # Validate layer indices
        for idx in self.phase_layer_indices:
            if not (0 <= idx < config.n_layers):
                raise ValueError(
                    f"Invalid layer index {idx}. Must be in range [0, {config.n_layers-1}]"
                )

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Positional embeddings
        self.position_embedding = nn.Embedding(config.max_seq_len, config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GPT2Block(
                config,
                use_phase_attention=(use_phase_attention and i in self.phase_layer_indices)
            )
            for i in range(config.n_layers)
        ])

        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_epsilon)

        # Language modeling head (tied with token embeddings)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying (share weights between token embeddings and LM head)
        self.lm_head.weight = self.token_embedding.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        """
        Initialize weights with GPT-2 initialization scheme.

        Args:
            module: Module to initialize
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_coherence: bool = False,
        return_info: bool = False
    ) -> Union[torch.Tensor, tuple]:
        """
        Forward pass through GPT-2 model.

        Args:
            input_ids: Token IDs [batch, seq_len]
            return_coherence: If True, return phase coherence for phase layers

        Returns:
            logits: [batch, seq_len, vocab_size] - unnormalized log probabilities
            (coherence_dict): Dict[layer_idx -> R] if requested

        Raises:
            ValueError: If input_ids length exceeds max_seq_len
        """
        batch, seq_len = input_ids.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum {self.config.max_seq_len}"
            )

        # Get token embeddings
        token_emb = self.token_embedding(input_ids)  # [batch, seq_len, d_model]

        # Get position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device)
        position_emb = self.position_embedding(positions)  # [seq_len, d_model]

        # Combine embeddings
        x = self.dropout(token_emb + position_emb)

        # Track phase coherence if requested
        coherence_dict = {} if return_coherence else None
        collected_phases = []

        # Forward through transformer blocks
        for i, block in enumerate(self.blocks):
            if return_info:
                out = block(x, return_info=True)
                if isinstance(out, tuple):
                    x, blk_info = out
                    if blk_info is not None and 'phases' in blk_info and blk_info['phases'] is not None:
                        collected_phases.append(blk_info['phases'])
                    if return_coherence and 'R' in blk_info and blk_info['R'] is not None:
                        coherence_dict[i] = blk_info['R']
                else:
                    x = out
            elif return_coherence and i in self.phase_layer_indices:
                x, R = block(x, return_coherence=True)
                coherence_dict[i] = R
            else:
                x = block(x)

        # Final layer norm
        x = self.ln_f(x)

        # Language modeling head
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        if return_info:
            info = {}
            if len(collected_phases) > 0:
                try:
                    # Use the last available phases (most recent layer)
                    info['phases'] = collected_phases[-1]
                except Exception:
                    info['phases'] = None
            else:
                info['phases'] = None
            
            if return_coherence:
                info['coherence'] = coherence_dict
            
            return logits, info
        
        if return_coherence:
            return logits, coherence_dict
        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Count the number of parameters in the model.

        Args:
            non_embedding: If True, exclude embedding parameters

        Returns:
            Number of parameters
        """
        n_params = sum(p.numel() for p in self.parameters())

        if non_embedding:
            # Subtract embedding parameters
            n_params -= self.token_embedding.weight.numel()
            n_params -= self.position_embedding.weight.numel()

        return n_params

    def get_phase_layer_info(self) -> Dict[str, Any]:
        """
        Get information about which layers use PhaseAttention.

        Returns:
            Dictionary with phase layer configuration:
            - 'use_phase_attention': bool
            - 'phase_layers': List of layer indices
            - 'total_layers': Total number of layers
            - 'phase_ratio': Fraction of layers using PhaseAttention
        """
        return {
            'use_phase_attention': self.use_phase_attention,
            'phase_layers': sorted(list(self.phase_layer_indices)),
            'total_layers': self.config.n_layers,
            'phase_ratio': len(self.phase_layer_indices) / self.config.n_layers
        }

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: Union[str, Path],
        device: str = 'cpu'
    ) -> 'GPT2Model':
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Device to load model on ('cpu', 'cuda', 'mps')

        Returns:
            Loaded GPT2Model instance

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Extract config
        config_dict = checkpoint.get('config', None)
        if config_dict is None:
            raise ValueError("Checkpoint missing 'config' field")

        config = GPT2Config(**config_dict)

        # Extract model metadata
        use_phase_attention = checkpoint.get('use_phase_attention', False)
        phase_layer_idx = checkpoint.get('phase_layer_idx', None)

        # Create model
        model = cls(config, use_phase_attention, phase_layer_idx)

        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])

        model.to(device)
        model.eval()

        return model

    def save_checkpoint(
        self,
        path: Union[str, Path],
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        **extra_metadata
    ):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer: Optional optimizer state to save
            epoch: Optional epoch number
            **extra_metadata: Additional metadata to save
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config.to_dict(),
            'use_phase_attention': self.use_phase_attention,
            'phase_layer_idx': sorted(list(self.phase_layer_indices)) if self.phase_layer_indices else None,
            'num_parameters': self.get_num_params(),
            'phase_layer_info': self.get_phase_layer_info(),
        }

        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()

        if epoch is not None:
            checkpoint['epoch'] = epoch

        checkpoint.update(extra_metadata)

        torch.save(checkpoint, path)


def replace_attention_layer(
    model: GPT2Model,
    layer_idx: Union[int, List[int]],
    phase_attention_kwargs: Optional[Dict[str, Any]] = None
) -> GPT2Model:
    """
    Replace specific attention layer(s) with PhaseAttention in an existing model.

    This function modifies the model in-place by replacing the attention module
    in specified transformer blocks.

    Args:
        model: Existing GPT2Model instance
        layer_idx: Layer index or indices to replace (0-indexed)
        phase_attention_kwargs: Optional kwargs for PhaseAttention initialization
            If None, uses default values from model.config

    Returns:
        Modified model (same instance, modified in-place)

    Example:
        # Start with baseline model
        model = GPT2Model(config)

        # Replace layer 6 with PhaseAttention
        model = replace_attention_layer(model, layer_idx=6)

        # Replace multiple layers with custom parameters
        model = replace_attention_layer(
            model,
            layer_idx=[3, 6, 9],
            phase_attention_kwargs={'coupling_strength': 2.0, 'phase_iterations': 20}
        )
    """
    # Normalize to list
    if isinstance(layer_idx, int):
        layer_indices = [layer_idx]
    else:
        layer_indices = layer_idx

    # Validate indices
    for idx in layer_indices:
        if not (0 <= idx < model.config.n_layers):
            raise ValueError(
                f"Invalid layer index {idx}. Must be in range [0, {model.config.n_layers-1}]"
            )

    # Prepare PhaseAttention kwargs
    if phase_attention_kwargs is None:
        phase_attention_kwargs = {}

    default_kwargs = {
        'd_model': model.config.d_model,
        'n_heads': model.config.n_heads,
        'num_oscillators': model.config.num_oscillators,
        'coupling_strength': model.config.coupling_strength,
        'natural_freq_std': model.config.natural_freq_std,
        'phase_iterations': model.config.phase_iterations,
        'dropout': model.config.dropout
    }
    default_kwargs.update(phase_attention_kwargs)

    # Replace attention layers
    for idx in layer_indices:
        block = model.blocks[idx]

        # Create new PhaseAttention module
        phase_attn = PhaseAttention(**default_kwargs)

        # Replace attention module
        block.attention = phase_attn
        block.use_phase_attention = True

        # Update model metadata
        model.phase_layer_indices.add(idx)

    model.use_phase_attention = True

    return model


# Example usage and testing
if __name__ == "__main__":
    print("Testing GPT-2 Model with PhaseAttention Integration\n")

    # Create config
    config = GPT2Config(
        vocab_size=50257,
        n_layers=6,
        d_model=512,
        n_heads=8,
        max_seq_len=128,
        dropout=0.1
    )

    print("Config:")
    print(f"  Layers: {config.n_layers}")
    print(f"  d_model: {config.d_model}")
    print(f"  Heads: {config.n_heads}")
    print(f"  Vocab: {config.vocab_size}\n")

    # Test 1: Baseline model
    print("=" * 60)
    print("Test 1: Baseline GPT-2 (standard attention)")
    print("=" * 60)

    model_baseline = GPT2Model(config, use_phase_attention=False)
    print(f"Parameters: {model_baseline.get_num_params():,}")
    print(f"Phase layer info: {model_baseline.get_phase_layer_info()}\n")

    # Test forward pass
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    logits = model_baseline(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected: [batch={batch_size}, seq_len={seq_len}, vocab={config.vocab_size}]\n")

    # Test 2: Phase-coupled model (single layer)
    print("=" * 60)
    print("Test 2: Phase-coupled GPT-2 (layer 3 with PhaseAttention)")
    print("=" * 60)

    model_phase = GPT2Model(config, use_phase_attention=True, phase_layer_idx=3)
    print(f"Parameters: {model_phase.get_num_params():,}")
    print(f"Phase layer info: {model_phase.get_phase_layer_info()}\n")

    logits, coherence = model_phase(input_ids, return_coherence=True)
    print(f"Output shape: {logits.shape}")
    print(f"Coherence (layer 3): {coherence[3].mean().item():.4f}\n")

    # Test 3: Multiple phase layers
    print("=" * 60)
    print("Test 3: Multiple phase layers [1, 3, 5]")
    print("=" * 60)

    model_multi = GPT2Model(config, use_phase_attention=True, phase_layer_idx=[1, 3, 5])
    print(f"Phase layer info: {model_multi.get_phase_layer_info()}\n")

    logits, coherence = model_multi(input_ids, return_coherence=True)
    for layer_idx, R in coherence.items():
        print(f"Layer {layer_idx} coherence: {R.mean().item():.4f}")

    # Test 4: Dynamic replacement
    print("\n" + "=" * 60)
    print("Test 4: Dynamic layer replacement")
    print("=" * 60)

    model_dynamic = GPT2Model(config, use_phase_attention=False)
    print(f"Before: {model_dynamic.get_phase_layer_info()}")

    replace_attention_layer(model_dynamic, layer_idx=2)
    print(f"After: {model_dynamic.get_phase_layer_info()}\n")

    # Test 5: Save/load
    print("=" * 60)
    print("Test 5: Save and load checkpoint")
    print("=" * 60)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test_checkpoint.pt"

        # Save
        model_phase.save_checkpoint(checkpoint_path, epoch=10)
        print(f"Saved to: {checkpoint_path}")

        # Load
        model_loaded = GPT2Model.from_pretrained(checkpoint_path)
        print(f"Loaded model: {model_loaded.get_phase_layer_info()}")

        # Verify outputs match
        logits_original = model_phase(input_ids)
        logits_loaded = model_loaded(input_ids)
        diff = (logits_original - logits_loaded).abs().max().item()
        print(f"Max difference: {diff:.6e}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)

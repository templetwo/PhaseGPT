"""
Evaluation Script for GPT-2 with PhaseAttention

Provides comprehensive evaluation tools for comparing baseline and phase-coupled
GPT-2 models:
- Perplexity computation on validation data
- Phase coherence measurement (R parameter) for phase models
- Text generation with sampling
- Comparative analysis between model variants

Usage:
    # Evaluate model on validation set
    python src/evaluate.py --checkpoint experiments/run1/checkpoints/best.pt \\
                          --data data/processed/val.txt

    # Generate text samples
    python src/evaluate.py --checkpoint experiments/run1/checkpoints/best.pt \\
                          --generate --num_samples 5 --max_length 100

    # Compare two models
    python src/evaluate.py --checkpoint experiments/baseline/best.pt \\
                          --compare_with experiments/phase/best.pt \\
                          --data data/processed/val.txt

    # Full evaluation (perplexity + generation + coherence)
    python src/evaluate.py --checkpoint experiments/run1/checkpoints/best.pt \\
                          --data data/processed/val.txt \\
                          --generate --num_samples 3 \\
                          --output_dir evaluation_results/
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import GPT2Model, GPT2Config


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Results from model evaluation."""

    # Model info
    model_name: str
    checkpoint_path: str
    num_parameters: int
    phase_layer_info: Dict[str, Any]

    # Perplexity metrics
    perplexity: Optional[float] = None
    avg_loss: Optional[float] = None
    num_tokens: Optional[int] = None

    # Phase coherence (if applicable)
    avg_coherence: Optional[float] = None
    coherence_per_layer: Optional[Dict[int, float]] = None

    # Generation samples
    generated_samples: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, path: Path):
        """Save results to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def __str__(self) -> str:
        """Pretty print results."""
        lines = [
            "=" * 70,
            "Evaluation Results",
            "=" * 70,
            f"Model: {self.model_name}",
            f"Checkpoint: {self.checkpoint_path}",
            f"Parameters: {self.num_parameters:,}",
            "",
            "Phase Configuration:",
            f"  Use Phase Attention: {self.phase_layer_info['use_phase_attention']}",
            f"  Phase Layers: {self.phase_layer_info['phase_layers']}",
            f"  Phase Ratio: {self.phase_layer_info['phase_ratio']:.2%}",
            "",
        ]

        if self.perplexity is not None:
            lines.extend([
                "Perplexity Metrics:",
                f"  Perplexity: {self.perplexity:.4f}",
                f"  Avg Loss: {self.avg_loss:.4f}",
                f"  Tokens Evaluated: {self.num_tokens:,}",
                "",
            ])

        if self.avg_coherence is not None:
            lines.extend([
                "Phase Coherence (R parameter):",
                f"  Average: {self.avg_coherence:.4f}",
                "  Per Layer:",
            ])
            for layer_idx, R in sorted(self.coherence_per_layer.items()):
                lines.append(f"    Layer {layer_idx}: {R:.4f}")
            lines.append("")

        if self.generated_samples:
            lines.extend([
                f"Generated Samples ({len(self.generated_samples)}):",
                "-" * 70,
            ])
            for i, sample in enumerate(self.generated_samples, 1):
                lines.append(f"\nSample {i}:")
                lines.append(sample)
                lines.append("-" * 70)

        lines.append("=" * 70)
        return "\n".join(lines)


class Evaluator:
    """
    Evaluator for GPT-2 models with PhaseAttention support.

    Handles:
    - Loading checkpoints and data
    - Computing perplexity
    - Measuring phase coherence
    - Generating text samples
    - Comparative analysis
    """

    def __init__(
        self,
        checkpoint_path: Path,
        device: str = 'auto',
        batch_size: int = 8
    ):
        """
        Initialize evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use ('cpu', 'cuda', 'mps', or 'auto')
            batch_size: Batch size for evaluation
        """
        self.checkpoint_path = checkpoint_path
        self.batch_size = batch_size

        # Auto-detect device
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Load model
        logger.info(f"Loading model from {checkpoint_path}")
        self.model = GPT2Model.from_pretrained(checkpoint_path, device=self.device)
        self.model.eval()

        logger.info(f"Model loaded: {self.model.get_num_params():,} parameters")
        logger.info(f"Phase info: {self.model.get_phase_layer_info()}")

    def load_text_data(self, data_path: Path) -> List[str]:
        """
        Load text data from file.

        Args:
            data_path: Path to text file (one example per line)

        Returns:
            List of text strings

        Raises:
            FileNotFoundError: If data file doesn't exist
        """
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(lines)} examples from {data_path}")
        return lines

    def tokenize(self, text: str) -> torch.Tensor:
        """
        Tokenize text to token IDs.

        Note: This is a placeholder. In practice, you'd use a proper tokenizer
        like GPT2Tokenizer from transformers library.

        Args:
            text: Input text string

        Returns:
            Token IDs tensor [seq_len]
        """
        # Placeholder: In real implementation, use proper tokenizer
        # For now, convert characters to ASCII values (toy tokenizer)
        token_ids = torch.tensor([ord(c) % self.model.config.vocab_size for c in text])
        return token_ids

    def compute_perplexity(
        self,
        data_path: Path,
        max_examples: Optional[int] = None
    ) -> Tuple[float, float, int]:
        """
        Compute perplexity on validation data.

        Perplexity = exp(average cross-entropy loss)

        Args:
            data_path: Path to validation data file
            max_examples: Optional limit on number of examples

        Returns:
            Tuple of (perplexity, avg_loss, num_tokens)
        """
        logger.info(f"Computing perplexity on {data_path}")

        # Load data
        texts = self.load_text_data(data_path)
        if max_examples:
            texts = texts[:max_examples]

        total_loss = 0.0
        total_tokens = 0

        self.model.eval()
        with torch.no_grad():
            for text in tqdm(texts, desc="Evaluating"):
                # Tokenize
                token_ids = self.tokenize(text)

                # Skip if too short or too long
                if len(token_ids) < 2:
                    continue
                if len(token_ids) > self.model.config.max_seq_len:
                    token_ids = token_ids[:self.model.config.max_seq_len]

                # Prepare input and target
                input_ids = token_ids[:-1].unsqueeze(0).to(self.device)
                target_ids = token_ids[1:].to(self.device)

                # Forward pass
                logits = self.model(input_ids)  # [1, seq_len, vocab_size]

                # Compute loss
                logits = logits.squeeze(0)  # [seq_len, vocab_size]
                loss = F.cross_entropy(logits, target_ids, reduction='sum')

                total_loss += loss.item()
                total_tokens += len(target_ids)

        # Compute metrics
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()

        logger.info(f"Perplexity: {perplexity:.4f}")
        logger.info(f"Avg Loss: {avg_loss:.4f}")
        logger.info(f"Tokens: {total_tokens:,}")

        return perplexity, avg_loss, total_tokens

    def measure_coherence(
        self,
        data_path: Path,
        max_examples: Optional[int] = 100
    ) -> Tuple[float, Dict[int, float]]:
        """
        Measure phase coherence (R parameter) for phase-coupled models.

        Args:
            data_path: Path to validation data
            max_examples: Number of examples to measure on

        Returns:
            Tuple of (avg_coherence, coherence_per_layer)

        Raises:
            ValueError: If model doesn't use PhaseAttention
        """
        if not self.model.use_phase_attention:
            raise ValueError("Model doesn't use PhaseAttention")

        logger.info("Measuring phase coherence (R parameter)")

        # Load data
        texts = self.load_text_data(data_path)
        if max_examples:
            texts = texts[:max_examples]

        # Track coherence per layer
        coherence_sums = {idx: 0.0 for idx in self.model.phase_layer_indices}
        num_samples = 0

        self.model.eval()
        with torch.no_grad():
            for text in tqdm(texts, desc="Measuring coherence"):
                # Tokenize
                token_ids = self.tokenize(text)

                if len(token_ids) < 2:
                    continue
                if len(token_ids) > self.model.config.max_seq_len:
                    token_ids = token_ids[:self.model.config.max_seq_len]

                input_ids = token_ids.unsqueeze(0).to(self.device)

                # Forward pass with coherence measurement
                _, coherence_dict = self.model(input_ids, return_coherence=True)

                # Accumulate coherence values
                for layer_idx, R in coherence_dict.items():
                    coherence_sums[layer_idx] += R.mean().item()

                num_samples += 1

        # Compute averages
        coherence_per_layer = {
            idx: total / num_samples
            for idx, total in coherence_sums.items()
        }

        avg_coherence = sum(coherence_per_layer.values()) / len(coherence_per_layer)

        logger.info(f"Average coherence: {avg_coherence:.4f}")
        for layer_idx, R in sorted(coherence_per_layer.items()):
            logger.info(f"  Layer {layer_idx}: {R:.4f}")

        return avg_coherence, coherence_per_layer

    def generate_text(
        self,
        prompt: str = "",
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = 50,
        num_samples: int = 1
    ) -> List[str]:
        """
        Generate text samples from the model.

        Args:
            prompt: Starting text (empty string for unconditional generation)
            max_length: Maximum length to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k tokens
            num_samples: Number of samples to generate

        Returns:
            List of generated text strings
        """
        logger.info(f"Generating {num_samples} samples (max_length={max_length})")

        samples = []
        self.model.eval()

        for i in range(num_samples):
            with torch.no_grad():
                # Tokenize prompt
                if prompt:
                    token_ids = self.tokenize(prompt).to(self.device)
                else:
                    # Start with random token
                    token_ids = torch.randint(
                        0, self.model.config.vocab_size, (1,), device=self.device
                    )

                generated = token_ids.tolist()

                # Generate tokens
                for _ in range(max_length):
                    # Get current sequence
                    input_ids = torch.tensor([generated], device=self.device)

                    # Truncate if too long
                    if input_ids.shape[1] > self.model.config.max_seq_len:
                        input_ids = input_ids[:, -self.model.config.max_seq_len:]

                    # Forward pass
                    logits = self.model(input_ids)  # [1, seq_len, vocab_size]

                    # Get logits for last token
                    logits = logits[0, -1, :] / temperature

                    # Apply top-k filtering
                    if top_k is not None:
                        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                        logits[indices_to_remove] = float('-inf')

                    # Sample from distribution
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    generated.append(next_token.item())

                # Decode (placeholder - use proper detokenizer in practice)
                text = ''.join([chr((tok % 128) + 32) for tok in generated])

                samples.append(text)

                logger.info(f"Sample {i+1}/{num_samples} generated ({len(generated)} tokens)")

        return samples

    def evaluate(
        self,
        data_path: Optional[Path] = None,
        generate: bool = False,
        num_samples: int = 3,
        max_length: int = 100,
        output_dir: Optional[Path] = None
    ) -> EvaluationResults:
        """
        Run full evaluation.

        Args:
            data_path: Optional path to validation data (for perplexity)
            generate: If True, generate text samples
            num_samples: Number of samples to generate
            max_length: Max length for generation
            output_dir: Optional directory to save results

        Returns:
            EvaluationResults object
        """
        results = EvaluationResults(
            model_name=self.checkpoint_path.stem,
            checkpoint_path=str(self.checkpoint_path),
            num_parameters=self.model.get_num_params(),
            phase_layer_info=self.model.get_phase_layer_info()
        )

        # Compute perplexity if data provided
        if data_path:
            try:
                perplexity, avg_loss, num_tokens = self.compute_perplexity(data_path)
                results.perplexity = perplexity
                results.avg_loss = avg_loss
                results.num_tokens = num_tokens

                # Measure coherence for phase models
                if self.model.use_phase_attention:
                    avg_coherence, coherence_per_layer = self.measure_coherence(data_path)
                    results.avg_coherence = avg_coherence
                    results.coherence_per_layer = coherence_per_layer

            except Exception as e:
                logger.error(f"Error during evaluation: {e}")

        # Generate samples if requested
        if generate:
            try:
                samples = self.generate_text(
                    num_samples=num_samples,
                    max_length=max_length
                )
                results.generated_samples = samples
            except Exception as e:
                logger.error(f"Error during generation: {e}")

        # Save results if output dir provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            results_path = output_dir / f"{results.model_name}_results.json"
            results.to_json(results_path)
            logger.info(f"Results saved to {results_path}")

        return results


def compare_models(
    checkpoint1: Path,
    checkpoint2: Path,
    data_path: Path,
    device: str = 'auto'
) -> Dict[str, EvaluationResults]:
    """
    Compare two models side-by-side.

    Args:
        checkpoint1: Path to first model checkpoint
        checkpoint2: Path to second model checkpoint
        data_path: Path to validation data
        device: Device to use

    Returns:
        Dictionary mapping model name to results
    """
    logger.info("=" * 70)
    logger.info("Model Comparison")
    logger.info("=" * 70)

    results = {}

    # Evaluate model 1
    logger.info(f"\nEvaluating Model 1: {checkpoint1.name}")
    evaluator1 = Evaluator(checkpoint1, device=device)
    results['model1'] = evaluator1.evaluate(data_path=data_path)

    # Evaluate model 2
    logger.info(f"\nEvaluating Model 2: {checkpoint2.name}")
    evaluator2 = Evaluator(checkpoint2, device=device)
    results['model2'] = evaluator2.evaluate(data_path=data_path)

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    for name, result in results.items():
        print(f"\n{name.upper()}: {result.model_name}")
        print(f"  Parameters: {result.num_parameters:,}")
        print(f"  Phase Layers: {result.phase_layer_info['phase_layers']}")

        if result.perplexity:
            print(f"  Perplexity: {result.perplexity:.4f}")

        if result.avg_coherence:
            print(f"  Avg Coherence: {result.avg_coherence:.4f}")

    # Compute differences
    if results['model1'].perplexity and results['model2'].perplexity:
        ppl_diff = results['model1'].perplexity - results['model2'].perplexity
        ppl_pct = (ppl_diff / results['model1'].perplexity) * 100

        print(f"\nPerplexity Difference:")
        print(f"  Absolute: {ppl_diff:+.4f}")
        print(f"  Relative: {ppl_pct:+.2f}%")
        print(f"  Winner: {'Model 2' if ppl_diff > 0 else 'Model 1'}")

    print("=" * 70)

    return results


def main():
    """Main entry point for evaluation script."""
    parser = argparse.ArgumentParser(
        description='Evaluate GPT-2 models with PhaseAttention',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate on validation set
  python src/evaluate.py --checkpoint experiments/run1/best.pt --data data/val.txt

  # Generate text samples
  python src/evaluate.py --checkpoint experiments/run1/best.pt --generate --num_samples 5

  # Compare two models
  python src/evaluate.py --checkpoint experiments/baseline/best.pt \\
                        --compare_with experiments/phase/best.pt \\
                        --data data/val.txt

  # Full evaluation with output
  python src/evaluate.py --checkpoint experiments/run1/best.pt \\
                        --data data/val.txt --generate \\
                        --output_dir results/
        """
    )

    # Required arguments
    parser.add_argument(
        '--checkpoint',
        type=Path,
        required=True,
        help='Path to model checkpoint'
    )

    # Data arguments
    parser.add_argument(
        '--data',
        type=Path,
        help='Path to validation data (for perplexity)'
    )

    # Generation arguments
    parser.add_argument(
        '--generate',
        action='store_true',
        help='Generate text samples'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=3,
        help='Number of samples to generate (default: 3)'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=100,
        help='Maximum generation length (default: 100)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (default: 1.0)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=50,
        help='Top-k sampling (default: 50)'
    )

    # Comparison arguments
    parser.add_argument(
        '--compare_with',
        type=Path,
        help='Path to second model for comparison'
    )

    # Output arguments
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='Directory to save evaluation results'
    )

    # System arguments
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda', 'mps'],
        help='Device to use (default: auto)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=8,
        help='Batch size for evaluation (default: 8)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.checkpoint.exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    if args.data and not args.data.exists():
        logger.error(f"Data file not found: {args.data}")
        sys.exit(1)

    if not args.data and not args.generate:
        logger.error("Must provide either --data or --generate")
        sys.exit(1)

    # Run comparison if requested
    if args.compare_with:
        if not args.compare_with.exists():
            logger.error(f"Comparison checkpoint not found: {args.compare_with}")
            sys.exit(1)

        if not args.data:
            logger.error("Must provide --data for comparison")
            sys.exit(1)

        results = compare_models(
            args.checkpoint,
            args.compare_with,
            args.data,
            device=args.device
        )

        # Save comparison results
        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            for name, result in results.items():
                result_path = output_dir / f"{name}_results.json"
                result.to_json(result_path)
                logger.info(f"Saved {name} results to {result_path}")

        sys.exit(0)

    # Single model evaluation
    evaluator = Evaluator(
        args.checkpoint,
        device=args.device,
        batch_size=args.batch_size
    )

    results = evaluator.evaluate(
        data_path=args.data,
        generate=args.generate,
        num_samples=args.num_samples,
        max_length=args.max_length,
        output_dir=args.output_dir
    )

    # Print results
    print("\n" + str(results))


if __name__ == "__main__":
    main()

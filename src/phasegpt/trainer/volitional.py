import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from typing import Optional, Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
from peft import get_peft_model, LoraConfig, TaskType

from phasegpt.core.architecture import ArchitectureConfig, verify_modules_exist

class VolitionalTrainer:
    """
    Custom Trainer for Volitional Silence and Intent-Driven Optimization.
    Supports QLoRA (4-bit) and MPS/CUDA acceleration.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        arch_config: ArchitectureConfig,
        lr: float = 2e-4,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 1
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.arch_config = arch_config
        self.lr = lr
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = self._get_device()
        
        # Initialize Architecture
        self._setup_architecture()
        
    def _get_device(self) -> torch.device:
        """Auto-detect device (MPS > CUDA > CPU)."""
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _setup_architecture(self):
        """Configure PEFT/LoRA based on ArchitectureConfig."""
        print(f"[VolitionalTrainer] Setting up architecture: {self.arch_config.model_type}")
        
        # Verify modules exist before applying adapters
        verify_modules_exist(self.model, self.arch_config)
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.arch_config.lora_rank,
            lora_alpha=self.arch_config.lora_alpha,
            lora_dropout=self.arch_config.lora_dropout,
            target_modules=self.arch_config.target_modules,
            modules_to_save=self.arch_config.modules_to_save,
            bias="none",
            inference_mode=False
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
        # Enable Gradient Checkpointing (Critical for VRAM/Memory saving)
        if hasattr(self.model, "gradient_checkpointing_enable"):
            print("[VolitionalTrainer] Enabling Gradient Checkpointing for memory efficiency.")
            self.model.gradient_checkpointing_enable()
            
        self.model.to(self.device)

    def train(self, train_dataset, num_epochs: int = 3):
        """
        Execute the Volitional Training Loop with Gradient Accumulation.
        """
        print(f"[VolitionalTrainer] Starting training on {self.device}", flush=True)
        
        # CRITICAL FIX FOR MAC OS / MPS:
        # num_workers=0: Run on main thread to avoid fork deadlocks
        # pin_memory=False: Disable pinned memory to avoid MPS resource contention/hangs
        dataloader = DataLoader(
            train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=False
        )
        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        
        # Mixed Precision Setup
        # CRITICAL FIX: Disable AMP on MPS to avoid GradScaler errors
        use_amp = self.device.type == "cuda"
        scaler = torch.cuda.amp.GradScaler() if use_amp else None
        
        self.model.train()
        
        global_step = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            optimizer.zero_grad() # Initialize gradients
            
            for step, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
                
                # Forward / Backward
                try:
                    if use_amp:
                        with torch.cuda.amp.autocast():
                            outputs = self.model(**batch)
                            loss = outputs.loss / self.gradient_accumulation_steps
                        scaler.scale(loss).backward()
                    else:
                        outputs = self.model(**batch)
                        loss = outputs.loss / self.gradient_accumulation_steps
                        loss.backward()
                    
                    # Gradient Accumulation Step
                    if (step + 1) % self.gradient_accumulation_steps == 0:
                        if use_amp:
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1
                        
                    total_loss += loss.item() * self.gradient_accumulation_steps
                    
                    if (step + 1) % (10 * self.gradient_accumulation_steps) == 0:
                        print(f"  Epoch {epoch+1} | Step {step} | Loss: {loss.item() * self.gradient_accumulation_steps:.4f}", flush=True)
                        
                except RuntimeError as e:
                    if "nan" in str(e).lower():
                        print(f"  [WARN] NaN loss detected at step {step}. Skipping batch.", flush=True)
                        optimizer.zero_grad()
                        continue
                    if "out of memory" in str(e).lower():
                        print(f"  [CRITICAL] OOM at step {step}. Clearing cache.")
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        if hasattr(torch.mps, "empty_cache"): torch.mps.empty_cache()
                        continue
                    raise e
                    
            avg_loss = total_loss / len(dataloader)
            print(f"[Epoch {epoch+1}] Average Loss: {avg_loss:.4f}")
            
    def save_adapters(self, output_dir: str):
        """Save only the trained adapters."""
        print(f"[VolitionalTrainer] Saving adapters to {output_dir}")
        self.model.save_pretrained(output_dir, safe_serialization=True)
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
import torch
from transformers import PreTrainedModel

class ArchitectureConfig(BaseModel):
    """
    Immutable configuration for Model Topology and PEFT strategy.
    Acts as the source of truth for VolitionalTrainer and vLLM inference.
    """
    model_type: str = Field(..., description="HuggingFace model type (e.g., 'llama', 'qwen2', 'gpt_neox')")
    model_name_or_path: str = Field(..., description="Path to base model or HuggingFace Hub ID")
    
    # PEFT / LoRA Configuration
    lora_rank: int = Field(16, description="Rank of LoRA adapters")
    lora_alpha: int = Field(32, description="Scaling factor for LoRA")
    lora_dropout: float = Field(0.05, description="Dropout probability for LoRA layers")
    
    # Explicit definition prevents "vLLM vs Transformers" naming conflicts
    target_modules: List[str] = Field(
        default_factory=list, 
        description="List of module names to apply LoRA adapters to (e.g., ['q_proj', 'v_proj'])"
    )
    
    # Critical for Volitional Silence (Special Token Training)
    modules_to_save: Optional[List[str]] = Field(
        default=None,
        description="List of modules to unfreeze entirely (e.g., ['embed_tokens', 'lm_head'])"
    )

    # Volitional Components
    volitional_head_dim: Optional[int] = Field(None, description="Dimension of the auxiliary volition head")

    @field_validator('target_modules', mode='before')
    def validate_target_modules(cls, v):
        if v is None:
            return []
        return v

    class Config:
        frozen = True  # Immutable after initialization


def detect_architecture(model: PreTrainedModel) -> ArchitectureConfig:
    """
    Introspects a loaded HuggingFace model to generate a valid ArchitectureConfig.
    Auto-populates target_modules and modules_to_save based on known lineages (Qwen vs Pythia).
    """
    model_type = getattr(model.config, "model_type", "unknown")
    model_name = getattr(model.config, "_name_or_path", "unknown-model")
    
    print(f"[Architecture Detection] Model Type: {model_type}")

    # Default values
    target_modules = []
    modules_to_save = []

    # 1. Qwen2 / Llama Lineage
    if model_type in ["qwen2", "llama", "mistral"]:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj", 
            "gate_proj", "up_proj", "down_proj"
        ]
        # Qwen2.5 typically unties embeddings, so we must save both
        modules_to_save = ["embed_tokens", "lm_head"]

    # 2. GPT-NeoX / Pythia Lineage
    elif model_type == "gpt_neox":
        target_modules = [
            "query_key_value", 
            "dense", 
            "dense_h_to_4h", 
            "dense_4h_to_h"
        ]
        modules_to_save = ["embed_in", "embed_out"]
    
    # 3. Fallback / Generic
    else:
        print(f"⚠ Warning: Unknown model type '{model_type}'. Using minimal default config.")
        target_modules = ["q_proj", "v_proj"]
        modules_to_save = ["embed_tokens", "lm_head"]

    return ArchitectureConfig(
        model_type=model_type,
        model_name_or_path=model_name,
        target_modules=target_modules,
        modules_to_save=modules_to_save
    )

def verify_modules_exist(model: PreTrainedModel, config: ArchitectureConfig) -> bool:
    """
    Verifies that the configured modules actually exist in the model.
    Returns True if valid, raises ValueError if critical modules are missing.
    """
    if not config.modules_to_save:
        return True

    print("[Module Verification]")
    all_modules = [name for name, _ in model.named_modules()]
    
    for module_name in config.modules_to_save:
        found = any(name.endswith(module_name) for name in all_modules)
        status = "✓" if found else "✗"
        print(f"  {status} Module '{module_name}'")
        
        if not found:
            raise ValueError(
                f"Critical module '{module_name}' not found in model architecture. "
                f"Cannot proceed with Volitional Training."
            )
            
    return True

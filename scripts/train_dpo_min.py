#!/usr/bin/env python3
import json, os, argparse, torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

def load_pairs(path):
    rows=[]
    with open(path) as f:
        for line in f:
            r=json.loads(line)
            chosen = r.get("chosen") or r.get("preferred")
            rejected= r.get("rejected")
            prompt = r.get("prompt","")
            if chosen and rejected:
                rows.append({"prompt":prompt,"chosen":chosen,"rejected":rejected})
    return Dataset.from_list(rows)

ap=argparse.ArgumentParser()
ap.add_argument("--base","-b",default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--data",default="data/preferences_v14_100pairs.jsonl")
ap.add_argument("--out","--output",default="checkpoints/v14/track_a/hybrid_sft_dpo/final")
ap.add_argument("--device",default="mps",choices=["cpu","cuda","mps"])
ap.add_argument("--beta",type=float,default=0.1)
ap.add_argument("--rank",type=int,default=16)
ap.add_argument("--steps",type=int,default=9)  # tiny, fast
ap.add_argument("--seed",type=int,default=42)
args=ap.parse_args()

tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype="auto")
if args.device=="mps" and torch.backends.mps.is_available(): model=model.to("mps")
elif args.device=="cuda" and torch.cuda.is_available(): model=model.to("cuda")
else: model=model.to("cpu")

peft_cfg = LoraConfig(r=args.rank, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","o_proj"])
model = get_peft_model(model, peft_cfg)

ds = load_pairs(args.data).train_test_split(test_size=0.16, seed=args.seed)
cfg = DPOConfig(output_dir=os.path.dirname(args.out),
                beta=args.beta, per_device_train_batch_size=8,
                per_device_eval_batch_size=8, max_steps=args.steps,
                save_safetensors=True, logging_steps=1, gradient_accumulation_steps=1)

trainer = DPOTrainer(model=model, args=cfg, processing_class=tok,
                     train_dataset=ds["train"], eval_dataset=ds["test"])
trainer.train()
os.makedirs(args.out, exist_ok=True)
trainer.model.save_pretrained(args.out)
tok.save_pretrained(args.out)
print("Saved adapter →", args.out)

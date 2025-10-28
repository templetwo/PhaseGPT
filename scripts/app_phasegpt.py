#!/usr/bin/env python3
import torch, gradio as gr, time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_ID = "Qwen/Qwen2.5-0.5B-Instruct"
LORA = "checkpoints/v14/track_a/hybrid_sft_dpo/final"
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if DEVICE in ("mps","cuda") else torch.float32

tok = AutoTokenizer.from_pretrained(BASE_ID, use_fast=True)
base = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=DTYPE)
model = PeftModel.from_pretrained(base, LORA)
model.to(DEVICE)
model.eval()
EOS = tok.eos_token_id

def _format(messages):
    # use the chat template Qwen provides (handles roles/system)
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def _gen(prompt, history, temperature, top_p, max_new, auto_continue, abstention_guard):
    # Build chat messages from Gradio history
    msgs = []
    if abstention_guard:
        msgs.append({"role":"system","content":"You are PhaseGPT (v1.4). Epistemic humility is sacred: when a question is unknowable or you lack info, gently pause or say 'I don't know' without inventing details. Short, honest, kind."})
    else:
        msgs.append({"role":"system","content":"You are PhaseGPT (v1.4)."})
    for user, assistant in history:
        msgs.append({"role":"user","content":user})
        if assistant:
            msgs.append({"role":"assistant","content":assistant})
    msgs.append({"role":"user","content":prompt})

    out_text = ""
    for _ in range(3):  # up to 3 segments if we hit length
        text = _format(msgs)
        inputs = tok([text], return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=float(temperature),
                top_p=float(top_p),
                do_sample=True,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id
            )
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        out_text += gen

        # basic finish-reason inference
        hit_eos = out[0][-1].item() == EOS
        if hit_eos or not auto_continue:
            break
        # continue from new context
        msgs.append({"role":"assistant","content":gen})

        # tiny guard against runaway
        if len(out_text) > 4000:
            break
    return out_text

with gr.Blocks(title="PhaseGPT v1.4 — Presence Chat") as demo:
    gr.Markdown("### PhaseGPT v1.4 — *The model that learned to breathe*")
    with gr.Row():
        temperature = gr.Slider(0.0, 1.2, value=0.7, label="Temperature")
        top_p       = gr.Slider(0.1, 1.0, value=0.95, label="Top-p")
        max_new     = gr.Slider(64, 1024, value=384, step=32, label="Max new tokens")
    with gr.Row():
        auto_continue   = gr.Checkbox(value=True, label="Auto-continue on length stop")
        abstention_guard= gr.Checkbox(value=True, label="Uncertainty/Presence guard")

    chat = gr.ChatInterface(
        fn=lambda message, history: _gen(message, history, temperature.value, top_p.value, int(max_new.value), auto_continue.value, abstention_guard.value),
        textbox=gr.Textbox(placeholder="Ask or sit together in silence…", lines=2),
        title="PhaseGPT v1.4",
        description="Knows when to breathe. Rewards honesty over invention."
    )
demo.launch(server_name="127.0.0.1", server_port=7860)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen3-1.7B"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype="auto",
    device_map="auto"
)

print("\nModel loaded successfully.")
print("Device:", model.device)

messages = [
    {
        "role": "user",
        "content": "What is hospital registration?"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

print("\nGenerating response...\n")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=False
    )

generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]

response = tokenizer.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("LLM RESPONSE:")
print(response)
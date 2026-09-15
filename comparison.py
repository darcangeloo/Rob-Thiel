from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import bnb_config

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb_config, device_map="auto")
peft_model = PeftModel.from_pretrained(base_model, "model/checkpoint-123")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

prompt = tokenizer.apply_chat_template([{"role": "user", "content": "Voglio avviare una startup, come posso iniziare a creare contenuti e catturare meglio i clienti?"}], tokenize=False, add_generation_prompt=True)

inputs = tokenizer(prompt, return_tensors="pt").to(base_model.device)

with peft_model.disable_adapter():
    output = peft_model.generate(**inputs, max_new_tokens=500, do_sample=False, repetition_penalty=1.1)
    
peft_model_output = peft_model.generate(**inputs, max_new_tokens=500, do_sample=False, repetition_penalty=1.1)

print(f"Qwen2.5 - 7B - Instruct: {tokenizer.decode(output[0], skip_special_tokens=True)}")
print(f"Rob Thiel Model: {tokenizer.decode(peft_model_output[0], skip_special_tokens=True)}")


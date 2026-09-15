from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from config import bnb_config

messages = []

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb_config, device_map="auto")

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

model = PeftModel.from_pretrained(base_model, "model/checkpoint-123")

while True:
    user_input = input("Tu: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    messages.append({"role": "user", "content": user_input})
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    output = model.generate(**inputs, max_new_tokens=200, do_sample=False, repetition_penalty=1.1)
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    response = full_text[len(tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()
    
    messages.append({"role": "assistant", "content": response})
    print(f"Rob Thiel: {response}")
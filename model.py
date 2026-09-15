from transformers import AutoModelForCausalLM
from peft import prepare_model_for_kbit_training
from config import bnb_config

qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", quantization_config=bnb_config, device_map="auto")
peft_model = prepare_model_for_kbit_training(qwen)
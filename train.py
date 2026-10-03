import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig 

device = "cpu" 
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("1. Model load kiya ja raha hai...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token 

model = AutoModelForCausalLM.from_pretrained(model_id).to(device)

print("2. Dataset load kiya ja raha hai...")
dataset = load_dataset("json", data_files="dataset.json", split="train")

def format_prompts(examples):
    formatted_texts = []
    for instruction, input_text, output in zip(examples["instruction"], examples["input"], examples["output"]):
        if input_text:
            prompt = f"User: {instruction}\nContext: {input_text}\nAssistant: {output}"
        else:
            prompt = f"User: {instruction}\nAssistant: {output}"
        formatted_texts.append(prompt)
    return {"text": formatted_texts}

dataset = dataset.map(format_prompts, batched=True)

print("3. LoRA Adapter set kiya ja raha hai...")
lora_config = LoraConfig(
    r=8, 
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], 
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 

print("4. Training start hone wali hai...")
training_args = SFTConfig(
    output_dir="./tinyllama-lora-output", 
    per_device_train_batch_size=1, 
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=1, 
    logging_steps=1, 
    use_cpu=True,
    dataset_text_field="text",
    max_length=128  # <--- FIX: Isey max_length kar diya
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args, 
)

print("Training Shuru!")
trainer.train()

print("Training Khatam! Model save kiya ja raha hai...")
trainer.save_model("./tinyllama-lora-output")
print("Process Complete. Folder me check karein.")
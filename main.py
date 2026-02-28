"""
Mainstream LLMs cannot, for the life of them, write in convincing Victorian styles, partly due to their being trained on so many modern sources, so I decided,
as a result of my wish for more classics written by women, to fine-tune a model on all of their nuances. 
Things to note: 
Several of the generations seen in the README file were done within the React+Vite+Flask app that I created using the .gguf file from this SLM, not directly in the Colab file. 
While creating that app, which I'll be putting up on Github soon, I tweaked the prompt and a few other things, so the code shown here may produce slightly different results. 
"""

#Imports and Other Installations - Take Care of in Separate Cells
# 1. Install pip upgrade
!pip install --upgrade pip
# ---- NEW CELL ---------
# More specific installations to ensure everything runs smoothly and every dependency has the correct version
!pip uninstall -y unsloth unsloth_zoo transformers datasets trl peft accelerate bitsandbytes xformers torch torchvision torchaudio
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl peft accelerate bitsandbytes datasets torchvision

# Standard imports
import torch
import requests
import re
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datasets import Dataset

# ==========================================
# 2. DATA ACQUISITION & DEEP CLEANING
# ==========================================

#All books acquired from Project Gutenberg for the full, raw text
urls = [
    "https://www.gutenberg.org/files/42671/42671-0.txt", # Pride and Prejudice
    "https://www.gutenberg.org/files/161/161-0.txt",     # Sense and Sensibility
    "https://www.gutenberg.org/files/158/158-0.txt",     # Emma
    "https://www.gutenberg.org/files/105/105-0.txt",     # Persuasion
    "https://www.gutenberg.org/files/121/121-0.txt",     # Northanger Abbey
    "https://www.gutenberg.org/files/141/141-0.txt",     # Mansfield Park
    "https://www.gutenberg.org/files/1212/1212-0.txt",   # Love and Freindship
    "https://www.gutenberg.org/files/9182/9182-0.txt",   # Villette
    "https://www.gutenberg.org/files/1260/1260-0.txt",   # Jane Eyre
    "https://www.gutenberg.org/files/969/969-0.txt",     # The Tenant of Wildfell Hall
    "https://www.gutenberg.org/files/768/768-0.txt",     # Wuthering Heights
    "https://www.gutenberg.org/files/514/514-0.txt",     # Little Women
    "https://www.gutenberg.org/files/84/84-0.txt",       # Frankenstein
    "https://www.gutenberg.org/files/145/145-0.txt",     # Middlemarch
    "https://www.gutenberg.org/files/767/767-0.txt",     # Agnes Grey
    "https://www.gutenberg.org/files/30486/30486-0.txt", # Shirley
    "https://www.gutenberg.org/files/1028/1028-0.txt",   # The Professor
    "https://www.gutenberg.org/files/6688/6688-0.txt",   # The Mill on the Floss
    "https://www.gutenberg.org/files/550/550-0.txt",     # Silas Marner
    "https://www.gutenberg.org/files/507/507-0.txt",     # Adam Bede
]

#Getting rid of certain title and chapter markers to prevent the SLM from using that in its responses
def clean_classic_text(text):
    start_markers = ["*** START OF THIS PROJECT", "*** START OF THE PROJECT"]
    end_markers = ["*** END OF THIS PROJECT", "*** END OF THE PROJECT"]
    start_idx = 0
    for m in start_markers:
        found = text.find(m)
        if found != -1:
            start_idx = text.find("\n", found) + 1
            break
    end_idx = len(text)
    for m in end_markers:
        found = text.find(m)
        if found != -1:
            end_idx = found
            break
    clean = text[start_idx:end_idx]
    clean = re.sub(r'CHAPTER\s+[IVXLCDM]+\.?', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'Chapter\s+\d+\.?', '', clean)
    clean = re.sub(r'\n\s*\n', '\n\n', clean)
    return clean.strip()

print("Downloading the Full Library...")
full_library_text = ""
for url in urls:
    r = requests.get(url)
    r.encoding = 'utf-8-sig'
    full_library_text += clean_classic_text(r.text) + "\n\n"

# ==========================================
# 3. LOAD LLAMA-3 & PREPARE
# ==========================================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32, 
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
)

def format_prompt(text_chunk):
    # A prompt made and formatted to be used during training to check the model's progress
    return {
        "text": f"### Instruction:\nCompose a narrative that embodies the wit, social precision, vocabulary, sentence structure, and emotional depth of 19th-century female literary classics, using the styles characterized by ornate vocabulary and winding sentences.\n\n### Input:\n{text_chunk[:200]}\n\n### Response:\n{text_chunk[200:]}"
    }

chunks = [full_library_text[i:i + 1100] for i in range(0, len(full_library_text), 1100)]
dataset = Dataset.from_dict({"raw_text": chunks})
dataset = dataset.map(lambda x: format_prompt(x["raw_text"]), remove_columns=["raw_text"])

# ==========================================
# 4. MONITORING CALLBACK
# ==========================================
class ProgressMonitor(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0 and state.global_step > 0:
            print(f"\n--- [LITERARY CHECK-IN: STEP {state.global_step}] ---")
            FastLanguageModel.for_inference(model)
            test_prompt = "The drawing room was silent, save for the crackle of the fire and the realization that"
            #I have since tested out multiple other prompts for this small language model, but for the purpose of something small that runs mostly as a test for the model within Colab, this one works reasonably well. 
            inputs = tokenizer([f"### Instruction:\nCompose a narrative that embodies the wit, social precision, vocabulary, sentence structure, and emotional depth of 19th-century female literary classics, using the styles characterized by ornate vocabulary and winding sentences.  \n\n### Input:\n{test_prompt}\n\n### Response:\n"], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.8) #I often opt for a temperature of 0.8 instead, as older linguistic conventions can be more treacherous for an SLM to navigate
            print(test_prompt,tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:\n")[-1])
            print("-" * 50)
            model.train() # Resume learning

# ==========================================
# 5. OPTIMIZED TRAINING (3 Epochs)
# ==========================================
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = 2048,
    callbacks = [ProgressMonitor()],
    args = TrainingArguments(
        num_train_epochs = 3,         
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_ratio = 0.1,           
        learning_rate = 1e-4,         
        lr_scheduler_type = "cosine", 
        weight_decay = 0.01,
        fp16 = not torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        output_dir = "classics_model_final",
    ),
)

trainer.train()

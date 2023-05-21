import os
import torch
import numpy as np
from datasets import load_dataset, concatenate_datasets
from transformers import DataCollatorForSeq2Seq
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, set_peft_model_state_dict

dataset_path = 'data/combined_verifier_data.json'
output_dir="flan-t5-xxl-lora-verifier"
val_size = 1000

model_id = "philschmid/flan-t5-xxl-sharded-fp16"

# Define LoRA Config
lora_config = LoraConfig(
 r=4,
 lora_alpha=16,
 target_modules=["q", "v"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_2_SEQ_LM
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16, load_in_8bit=True)

# Add Special Token for Training
new_tokens = ["<S>"]
tokenizer.add_tokens(new_tokens)
model.resize_token_embeddings(len(tokenizer))

# prepare int-8 model for training
model = prepare_model_for_int8_training(model)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

resume_from_checkpoint = False

if resume_from_checkpoint:
    checkpoint_path = "flan-t5-xxl-lora-veriifer/checkpoint-7200/"
    # Check the available weights and load them
    checkpoint_name = os.path.join(checkpoint_path, "pytorch_model.bin")
    adapters_weights = torch.load(checkpoint_name, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    set_peft_model_state_dict(model, adapters_weights)
    print("----------checkpoint loaded----------")

data = load_dataset("json", data_files=dataset_path)
dataset = data["train"].train_test_split(test_size=val_size, shuffle=True, seed=42)

# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["input"], truncation=True), batched=True, remove_columns=["input", "output"])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 90 percentile of max length for better utilization
max_source_length = int(np.percentile(input_lenghts, 100))
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["output"], truncation=True), batched=True, remove_columns=["input", "output"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 95 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 100))
print(f"Max target length: {max_target_length}")

def preprocess_function(sample, padding="max_length"):
    
    # add prefix to the input for t5
    inputs = []
    for i in range(len(sample["input"])):
        input = sample["instruction"][i] + ' ' + sample["input"][i]
        inputs.append(input)

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["output"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["input", "output", "instruction"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# save datasets to disk for later easy loading
# tokenized_dataset["train"].save_to_disk("data/train")
# tokenized_dataset["test"].save_to_disk("data/eval")

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=16,
    #auto_find_batch_size=True,
    warmup_steps=100,
    learning_rate=3e-4,
    num_train_epochs=2,
    optim="adamw_torch",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=1,
    evaluation_strategy="steps",
    logging_steps=5,
    # logging_strategy="steps",
    # logging_dir=f"{output_dir}/logs",
    # report_to="tensorboard",
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

# train model
trainer.train()


# save our LoRA model & tokenizer results
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# if you want to save the base model to call
# trainer.model.base_model.save_pretrained(output_dir)

# if you want to push it to hub
trainer.model.push_to_hub(output_dir)

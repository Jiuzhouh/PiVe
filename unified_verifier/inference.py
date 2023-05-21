import torch
import gradio as gr
from tqdm import tqdm
from datasets import load_dataset
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load peft config for pre-trained checkpoint etc.
peft_model_id = "jiuzhouh/flan-t5-xxl-lora-verifier"
config = PeftConfig.from_pretrained(peft_model_id)

# load base LLM model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

# Add Special Token for Inference
new_tokens = ["<S>"]
tokenizer.add_tokens(new_tokens)

# Load the Lora model
model = PeftModel.from_pretrained(model, peft_model_id, torch_dtype=torch.float16, device_map="auto")
model.eval()

print("Peft model loaded")

def evaluate(instruction, input, max_new_tokens=256, temperature=0.1, top_p=0.9, top_k=40, num_beams=4, num_return_sequences=1):
    model_input = instruction + ' ' + input
    input_ids = tokenizer(model_input, return_tensors="pt", truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, top_k=top_k, num_beams=num_beams, num_return_sequences=num_return_sequences)
    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]

    return output

gradio = False
generate_whole_file = True

if gradio:
    gr.Interface(
            fn=evaluate,
            inputs=[
                gr.components.Textbox(
                    lines=2,
                    label="Instruction",
                    placeholder="none",
                ),
                gr.components.Textbox(lines=2, label="Input", placeholder="none"),
                gr.components.Slider(
                    minimum=1, maximum=2000, step=1, value=256, label="Max tokens"
                ),
                gr.components.Slider(
                    minimum=0, maximum=1, value=0.1, label="Temperature"
                ),
                gr.components.Slider(
                    minimum=0, maximum=1, value=0.75, label="Top p"
                ),
                gr.components.Slider(
                    minimum=0, maximum=100, step=1, value=40, label="Top k"
                ),
                gr.components.Slider(
                    minimum=1, maximum=4, step=1, value=4, label="Beams"
                ),
            ],
            outputs=[
                gr.inputs.Textbox(
                    lines=5,
                    label="Output",
                )
            ],
            title="🦙🌲 FLAN-T5-XXL-LoRA",
            description="FLAN-T5-XXL-LoRA is a 11B-parameter T5-XXL model finetuned to follow instructions."
        ).queue().launch(server_name="0.0.0.0", share=True)

elif generate_whole_file:
    dataset_path = 'data/verifier_input.json'
    data = load_dataset("json", data_files=dataset_path)
    samples = data['train']
    with open('output/verifier_output.txt', 'a') as output_file:
        for sample in tqdm(samples):
            output = evaluate(sample['instruction'], sample['input'])
            output_file.write(str(output).strip() + '\n')

else:
    dataset_path = 'data/combined_verifier_data.json'
    test_size = 20
    data = load_dataset("json", data_files=dataset_path)
    dataset = data['train'].train_test_split(test_size=test_size, shuffle=True, seed=42)
    samples = dataset['test']

    for sample in samples:
        output = evaluate(sample['instruction'], sample['input'])
        print(f"Instruction: \n{sample['instruction']}\n{'---'* 20}")
        print(f"Input: \n{sample['input']}\n{'---'* 20}")
        print(f"Output:\n{output}\n{'---'* 20}")
        print(f"Reference: \n{sample['output']}\n{'---'* 20}\n{'---'* 20}")

    

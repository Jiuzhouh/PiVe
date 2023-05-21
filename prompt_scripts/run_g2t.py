import json
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

    
text = []
with open('data/GenWiki_source_graph/Iteration1/train.source', 'r') as f:
    for l in f.readlines():
        text.append(l.strip())

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-xl").cuda()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

with open("outputs/zero_shot_genwiki_results_all/flan_t5_xl_generated_train_I1.target","a") as output_file:
    for i in tqdm(range(len(text))):
        prompt = "Transform the semantic graph into a description. Semantic graph: " + text[i]
        inputs = tokenizer.encode(prompt, max_length=1024, truncation=False, return_tensors="pt").cuda()
        outputs = model.generate(inputs, max_length=1024)
        pred = tokenizer.decode(outputs[0].cpu(), skip_special_tokens=True)
        output_file.write(pred.strip() + '\n')


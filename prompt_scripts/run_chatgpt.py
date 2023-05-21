import os
import openai

openai.api_key = "API-KEY"

import time
import json
import functools
from tqdm import tqdm

text = []
with open('GPT3.5_result_KELM/test.target', 'r') as f:
    for l in f.readlines():
        text.append(l.strip())
#with open('GPT3.5_result_WebNLG/test.target', 'r') as f:
#    for l in f.readlines():
#        text.append(l.strip())
# with open('GPT3.5_result_GenWiki/test.target', 'r') as f:
#     for l in f.readlines():
#         text.append(l.strip())

@functools.lru_cache()
def get_chatgpt_completion(content):
    got_result = False
    while not got_result:
        try:
            response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages = [{"role": "user", "content": content}],
        temperature=0
        )
            got_result = True
        except Exception:
            time.sleep(1)
    return response["choices"][0][ "message"]["content"]

iteration1 = True

if iteration1:
    with open("GPT3.5_result_KELMs/Iteration1/test_generated_graphs.txt", "a") as output_file:
        for i in tqdm(range(len(text))):
            #WebNLG & GenWiki
            #prompt = "Transform the text into a semantic graph.\nExample:\nText: Sportpark De Toekomst is located in Ouder-Amstel, Netherlands. It is owned and operated by AFC Ajax N.V. and their tenants include the Ajax Youth Academy.\nSemantic graph: [[\"Sportpark De Toekomst\", \"location\", \"Ouder-Amstel\"], [\"Sportpark De Toekomst\", \"country\", \"Netherlands\"], [\"Sportpark De Toekomst\", \"owner\", \"AFC Ajax N.V.\"], [\"Sportpark De Toekomst\", \"operator\", \"AFC Ajax N.V.\"], [\"Sportpark De Toekomst\", \"tenant\", \"Ajax Youth Academy\"]]\nText: " + text[i] + "\nSemantic graph:"
            #KELM
            prompt = "Transform the text into a semantic graph. \nExample: \nText: Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.\nSemantic Graph: [[\"Shotgate Thickets\", \"instance of\", \"Nature reserve\"], [\"Shotgate Thickets\", \"country\", \"United Kingdom\"], [\"Shotgate Thickets\", \"operator\", \"Essex Wildlife Trust\"]]\nText: " + text[i] + "\nSemantic graph:"
            #KELM 6 shots
            # prompt="Transform the text into a semantic graph.\nExample 1:\nText: Grzegorz Piechna is a member of the FC Spartak Varna soccer team.\nSemantic graph: [[\"Grzegorz Piechna\", \"member of sports team\", \"FC Spartak Varna\"]]\n\nExample 2:\nText: The home ground of Washburn Ichabods football is Yager Stadium at Moore Bowl in the United States.\nSemantic graph: [[\"Washburn Ichabods football\", \"country\", \"United States\"], [\"Washburn Ichabods football\", \"home venue\", \"Yager Stadium at Moore Bowl\"]]\n\nExample 3:\nText: Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust.\nSemantic graph: [[\"Shotgate Thickets\", \"instance of\", \"Nature reserve\"], [\"Shotgate Thickets\", \"country\", \"United Kingdom\"], [\"Shotgate Thickets\", \"operator\", \"Essex Wildlife Trust\"]]\n\nExample 4:\nText: Pierre-François Hugues d'Hancarville is a human being who speaks French.\nSemantic graph: [[\"Pierre-François Hugues d'Hancarville\", \"languages spoken, written or signed\", \"French language\"], [\"Pierre-François Hugues d'Hancarville\", \"writing language\", \"French language\"], [\"Pierre-François Hugues d'Hancarville\", \"native language\", \"French language\"], [\"Pierre-François Hugues d'Hancarville\", \"instance of\", \"Human\"]]\n\nExample 5:\nText: Yves Ker Ambrun, whose given name is Yves, is a human from France. He works in Darmstadt and his name is Yves.\nSemantic graph: [[\"Yves Ker Ambrun\", \"country of citizenship\", \"France\"], [\"Yves Ker Ambrun\", \"sex or gender\", \"male\"], [\"Yves Ker Ambrun\", \"given name\", \"Yves ( given name )\"], [\"Yves Ker Ambrun\", \"work location\", \"Darmstadt\"], [\"Yves Ker Ambrun\", \"instance of\", \"Human\"]]\n\nExample 6:\nText: Kevin Laidlaw was born in Nightcaps, New Zealand on 9 August 1934. He attended St Kevin's College in Oamaru. He was a rugby union player who represented New Zealand in the men's team.\nSemantic graph: [\"Kevin Laidlaw\", \"occupation\", \"rugby union player\"], [\"Kevin Laidlaw\", \"date of birth\", \"09 August 1934\"], [\"Kevin Laidlaw\", \"place of birth\", \"Nightcaps\"], [\"Kevin Laidlaw\", \"country of citizenship\", \"New Zealand\"], [\"Kevin Laidlaw\", \"sex or gender\", \"male\"], [\"Kevin Laidlaw\", \"educated at\", \"St Kevin 's College , Oamaru\"]]\n\nText: " + text[i] + "\nSemantic graph:"
            response = get_chatgpt_completion(prompt)
            output_file.write(response.strip().replace('\n','') + '\n')
		
else:
    missing_triples = []
    with open('GPT3.5_result_KELM_flan/Iteration3/verifier_result.txt', 'r') as f:
        for l in f.readlines():
            missing_triples.append(l.strip())

    results_old = []
    with open("GPT3.5_result_KELM_flan/Iteration3/test_generated_graphs.txt", 'r') as f:
        for line in f.readlines():
            results_old.append(line.strip())

    with open("GPT3.5_result_KELM_flan/Iteration4/test_generated_graphs.txt", "a") as output_file:
        for i in tqdm(range(len(text))):
            if missing_triples[i] != 'Correct':
                #WebNLG & GenWiki
                #prompt = "Transform the text into a semantic graph and also add the given triples to the generated semantic graph.\nExample:\nText: Sportpark De Toekomst is located in Ouder-Amstel, Netherlands. It is owned and operated by AFC Ajax N.V. and their tenants include the Ajax Youth Academy. \nTriples: [\"Sportpark De Toekomst\", \"country\", \"Netherlands\"], [\"Sportpark De Toekomst\", \"operator\", \"AFC Ajax N.V.\"]\nSemantic graph: [[\"Sportpark De Toekomst\", \"location\", \"Ouder-Amstel\"], [\"Sportpark De Toekomst\", \"country\", \"Netherlands\"], [\"Sportpark De Toekomst\", \"owner\", \"AFC Ajax N.V.\"], [\"Sportpark De Toekomst\", \"operator\", \"AFC Ajax N.V.\"], [\"Sportpark De Toekomst\", \"tenant\", \"Ajax Youth Academy\"]]\nText: " + text[i] + "\nTriples: " + missing_triples[i] + "\nSemantic graph:"
                #KELM
                prompt = "Transform the text into a semantic graph and also add the given triples to the generated semantic graph.\nExample:\nText: Shotgate Thickets is a nature reserve in the United Kingdom operated by the Essex Wildlife Trust. \nTriples: [\"Shotgate Thickets\", \"instance of\", \"Nature reserve\"], [\"Shotgate Thickets\", \"country\", \"United Kingdom\"]\nSemantic graph: [[\"Shotgate Thickets\", \"instance of\", \"Nature reserve\"], [\"Shotgate Thickets\", \"country\", \"United Kingdom\"], [\"Shotgate Thickets\", \"operator\", \"Essex Wildlife Trust\"]]\nText: " + text[i] + "\nTriples: " + missing_triples[i] + "\nSemantic graph:"
                response = get_chatgpt_completion(prompt)
                output_file.write(response.strip().replace('\n','') + '\n')
            else:
                output_file.write(results_old[i].strip() + '\n')


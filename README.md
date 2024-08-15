# PiVe
This is the official code for the ACL 2024 paper: [PiVe: Prompting with Iterative Veriﬁcation Improving Graph-based Generative Capability of LLMs](https://aclanthology.org/2024.findings-acl.400.pdf).

## Files Introduction
1. `GenWiki-HIQ` is the created dataset using verifier module, which contains 110K parallel graph-text pairs.
2. `data_processing_script` contains `data_process.ipynb` to create the training data for the verifier module and test data for each iteration.
3. `datasets` contains the used kelm-sub and webnlg+2020 datasets. `pive_verifier_training_data.zip` contains the generated verifier training data for single verifier module and unified verifier module, which can be directly used to train the verifier modules.
4. `graph_evaluation` contains the graph evaluation metrics.
5. `prompt_scripts`contains the sctipts to prompt LLMs.
6. `single_verifier` contains the training sctipt for single verifier using T5-Large.
7. `unified_verifier` contains the training sctipt for unified verifier using insturction-tuning on Flan-T5-XXL.

## Clarification and Guidance 
For the file "data/only_one_error_webnlg/train.source" which is the training data for the verifier module, you need to use the first section of our provided data_process.ipynb to manually generate. We also upload the generated verifier training data in `pive_verifier_training_data.zip` for your convenience.

For the file "GPT3.5_result_KELM/test.target" in `run_chatgpt.py`, it is the same as the file which path is `datasets/kelm_sub/test.target`. You can just copy it to a folder like `GPT3.5_result_KELM` or use your own folder name, and put the corresponding file path in `run_chatgpt.py`. Then you can run the `run_chatgpt.py` to prompt LLMs for graph generation. After getting the results from LLMs, you need to use our `data_process.ipynb` to create the input for the single/unified verifier module from the generated graph. Then you can feed the input to the trained verifier module to predict the missing triple. For subsequent iterations, remember to set `iteration1 = False` in the `run_chatgpt.py` when prompting the LLMs.

## Citation
```
@inproceedings{han-etal-2024-pive,
    title = "{P}i{V}e: Prompting with Iterative Verification Improving Graph-based Generative Capability of {LLM}s",
    author = "Han, Jiuzhou  and
      Collier, Nigel  and
      Buntine, Wray  and
      Shareghi, Ehsan",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand and virtual meeting",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.400",
    pages = "6702--6718",
    abstract = "Large language models (LLMs) have shown great abilities of solving various natural language tasks in different domains. Due to the training objective of LLMs and their pre-training data, LLMs are not very well equipped for tasks involving structured data generation. We propose a framework, Prompting with Iterative Verification (PiVe), to improve graph-based generative capability of LLMs. We show how a small language model could be trained to act as a verifier module for the output of an LLM(i.e., ChatGPT, GPT-4), and to iteratively improve its performance via fine-grained corrective instructions. We also show how the verifier module could apply iterative corrections offline for a more cost-effective solution to the text-to-graph generation task. Experiments on three graph-based datasets show consistent improvement gained via PiVe. Additionally, we create GenWiki-HIQ and highlight that the verifier module can be used as a data augmentation tool to help improve the quality of automatically generated parallel text-graph datasets.",
}
```

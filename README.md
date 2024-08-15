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
@misc{han2023pive,
      title={PiVe: Prompting with Iterative Verification Improving Graph-based Generative Capability of LLMs}, 
      author={Jiuzhou Han and Nigel Collier and Wray Buntine and Ehsan Shareghi},
      year={2023},
      eprint={2305.12392},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

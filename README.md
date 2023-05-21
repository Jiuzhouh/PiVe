# PiVe
This is the official code for the paper: [PiVe: Prompting with Iterative Veriﬁcation Improving Graph-based Generative Capability of LLMs].

## Files Introduction
1. `GenWiki-HIQ` is the created dataset using verifier module, which contains 110K parallel graph-text pairs.
2. `data_processing_script` contains `data_process.ipynb` to create the training data for the verifier module and test data for each iteration.
3. `datasets` contains the used kelm-sub and webnlg+2020 datasets.
4. `graph_evaluation` contains the graph evaluation metrics.
5. `prompt_scripts`contains the sctipts to prompt LLMs.
6. `single_verifier` contains the training sctipt for single verifier using T5-Large.
7. `unified_verifier` contains the training sctipt for unified verifier using insturction-tuning on Flan-T5-XXL.

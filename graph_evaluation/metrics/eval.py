import ast
import argparse
from graph_matching import split_to_edges, get_tokens, get_bleu_rouge, get_bert_score, get_ged, get_triple_match_f1, get_graph_match_accuracy


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", default=None, type=str, required=True)
    parser.add_argument("--gold_file", default=None, type=str, required=True)

    args = parser.parse_args()
    
    gold_graphs = []
    with open(args.gold_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            gold_graphs.append(ast.literal_eval(line.strip()))
			
    pred_graphs = []
    with open(args.pred_file, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            #print(line)
            pred_graphs.append(ast.literal_eval(line.strip()))
			
    assert len(gold_graphs) == len(pred_graphs)
	
    # Evaluate Triple Match F1 Score
    triple_match_f1 = get_triple_match_f1(gold_graphs, pred_graphs)  

    # Evaluate Graph Match Accuracy
    graph_match_accuracy = get_graph_match_accuracy(pred_graphs, gold_graphs)

	# Compute GED
    overall_ged = 0.
    for (gold, pred) in zip(gold_graphs, pred_graphs):
        ged = get_ged(gold, pred)
        overall_ged += ged

	# Evaluate for Graph Matching
    gold_edges = split_to_edges(gold_graphs)
    pred_edges = split_to_edges(pred_graphs)
	
    gold_tokens, pred_tokens = get_tokens(gold_edges, pred_edges)

    precisions_rouge, recalls_rouge, f1s_rouge, precisions_bleu, recalls_bleu, f1s_bleu = get_bleu_rouge(
        gold_tokens, pred_tokens, gold_edges, pred_edges)

    precisions_BS, recalls_BS, f1s_BS = get_bert_score(gold_edges, pred_edges)

    print(f'Triple Match F1 Score: {triple_match_f1:.4f}\n')
    print(f'Graph Match F1 Score: {graph_match_accuracy:.4f}\n')
    
    print(f'G-BLEU Precision: {precisions_bleu.sum() / len(gold_graphs):.4f}')
    print(f'G-BLEU Recall: {recalls_bleu.sum() / len(gold_graphs):.4f}')
    print(f'G-BLEU F1: {f1s_bleu.sum() / len(gold_graphs):.4f}\n')

    print(f'G-Rouge Precision: {precisions_rouge.sum() / len(gold_graphs):.4f}')
    print(f'G-Rouge Recall Score: {recalls_rouge.sum() / len(gold_graphs):.4f}')
    print(f'G-Rouge F1 Score: {f1s_rouge.sum() / len(gold_graphs):.4f}\n')

    print(f'G-BertScore Precision Score: {precisions_BS.sum() / len(gold_graphs):.4f}')
    print(f'G-BertScore Recall Score: {recalls_BS.sum() / len(gold_graphs):.4f}')
    print(f'G-BertScore F1 Score: {f1s_BS.sum() / len(gold_graphs):.4f}\n')

    print(f'Graph Edit Distance (GED): {overall_ged / len(gold_graphs):.4f}\n')

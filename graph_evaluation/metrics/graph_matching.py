import numpy as np
from rouge_score import rouge_scorer
from bert_score import score as score_bert
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from scipy.optimize import linear_sum_assignment
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import re
import networkx as nx
from sklearn import preprocessing
from sklearn.metrics import precision_score, recall_score, f1_score

def modify_graph(original_graph):
    modified_graph = []
    for x in original_graph:
        modified_graph.append([str(t).lower().strip() for t in x])
    return modified_graph

def get_triple_match_f1(gold_graphs, pred_graphs):
    new_gold_graphs = [modify_graph(graph) for graph in gold_graphs]
    new_pred_graphs = [modify_graph(graph) for graph in pred_graphs]
    new_gold_graphs_list = [[str(string).lower() for string in sublist] for sublist in new_gold_graphs]
    new_pred_graphs_list = [[str(string).lower() for string in sublist] for sublist in new_pred_graphs]
    #First get all the classes by combining the triples in the pred_graphs and gold_graphs
    allclasses = new_pred_graphs_list + new_gold_graphs_list
    allclasses = [item for items in allclasses for item in items]
    allclasses = list(set(allclasses))

    lb = preprocessing.MultiLabelBinarizer(classes=allclasses)
    mcbin = lb.fit_transform(new_pred_graphs_list)
    mrbin = lb.fit_transform(new_gold_graphs_list)

    precision = precision_score(mrbin, mcbin, average='micro')
    recall = recall_score(mrbin, mcbin, average='micro')
    f1 = f1_score(mrbin, mcbin, average='micro')

    # print('Full triple scores')
    # print('-----------------------------------------------------------------')
    # print('Precision: ' + str(precision) + ' Recall: ' + str(recall) + '\nF1: ' + str(f1))
    return f1

def get_triple_match_accuracy(pred_graph, gold_graph):
    pred = modify_graph(pred_graph)
    gold = modify_graph(gold_graph)    
    matchs = 0
    for x in pred:
        if x in gold:
            matchs += 1
    acc = matchs/len(pred)
    return acc

def get_graph_match_accuracy(pred_graphs, gold_graphs):
    matchs = 0
    for pred, gold in zip(pred_graphs, gold_graphs):
        g1 = nx.DiGraph()
        g2 = nx.DiGraph()

        for edge in gold:
            g1.add_node(str(edge[0]).lower().strip(), label=str(edge[0]).lower().strip())
            g1.add_node(str(edge[2]).lower().strip(), label=str(edge[2]).lower().strip())
            g1.add_edge(str(edge[0]).lower().strip(), str(edge[2]).lower().strip(), label=str(edge[1]).lower().strip())

        for edge in pred:
            if len(edge) == 2:
                edge.append('NULL')
            elif len(edge) == 1:
                edge.append('NULL')
                edge.append('NULL')
            g2.add_node(str(edge[0]).lower().strip(), label=str(edge[0]).lower().strip())
            g2.add_node(str(edge[2]).lower().strip(), label=str(edge[2]).lower().strip())
            g2.add_edge(str(edge[0]).lower().strip(), str(edge[2]).lower().strip(), label=str(edge[1]).lower().strip())

        if nx.is_isomorphic(g1, g2, edge_match=lambda x, y: x == y):
            matchs += 1
    acc = matchs/len(pred_graphs)
    return acc

def get_tokens(gold_edges, pred_edges):
    nlp = English()
    tokenizer = Tokenizer(nlp.vocab, infix_finditer=re.compile(r'''[;]''').finditer)

    gold_tokens = []
    pred_tokens = []

    for i in range(len(gold_edges)):
        gold_tokens_edges = []
        pred_tokens_edges = []

        for sample in tokenizer.pipe(gold_edges[i]):
            gold_tokens_edges.append([j.text for j in sample])
        for sample in tokenizer.pipe(pred_edges[i]):
            pred_tokens_edges.append([j.text for j in sample])
        gold_tokens.append(gold_tokens_edges)
        pred_tokens.append(pred_tokens_edges)

    return gold_tokens, pred_tokens


def split_to_edges(graphs):
    processed_graphs = []
    for graph in graphs:
        #print(graph)
        processed_graphs.append([";".join(triple).lower().strip() for triple in graph])
    return processed_graphs


def get_bert_score(all_gold_edges, all_pred_edges):
    references = []
    candidates = []

    ref_cand_index = {}
    for (gold_edges, pred_edges) in zip(all_gold_edges, all_pred_edges):
        for (i, gold_edge) in enumerate(gold_edges):
            for (j, pred_edge) in enumerate(pred_edges):
                references.append(gold_edge)
                candidates.append(pred_edge)
                ref_cand_index[(gold_edge, pred_edge)] = len(references) - 1

    _, _, bs_F1 = score_bert(cands=candidates, refs=references, lang='en', idf=False)
    print("Computed bert scores for all pairs")

    precisions, recalls, f1s = [], [], []
    for (gold_edges, pred_edges) in zip(all_gold_edges, all_pred_edges):
        score_matrix = np.zeros((len(gold_edges), len(pred_edges)))
        for (i, gold_edge) in enumerate(gold_edges):
            for (j, pred_edge) in enumerate(pred_edges):
                score_matrix[i][j] = bs_F1[ref_cand_index[(gold_edge, pred_edge)]]

        row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

        sample_precision = score_matrix[row_ind, col_ind].sum() / len(pred_edges)
        sample_recall = score_matrix[row_ind, col_ind].sum() / len(gold_edges)

        precisions.append(sample_precision)
        recalls.append(sample_recall)
        f1s.append(2 * sample_precision * sample_recall / (sample_precision + sample_recall))

    return np.array(precisions), np.array(recalls), np.array(f1s)


# Note: These graph matching metrics are computed by considering each graph as a set of edges and each edge as a
# sentence
def get_bleu_rouge(gold_tokens, pred_tokens, gold_sent, pred_sent):
    scorer_rouge = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rouge3', 'rougeL'], use_stemmer=True)

    precisions_bleu = []
    recalls_bleu = []
    f1s_bleu = []

    precisions_rouge = []
    recalls_rouge = []
    f1s_rouge = []

    for graph_idx in range(len(gold_tokens)):
        score_bleu = np.zeros((len(pred_tokens[graph_idx]), len(gold_tokens[graph_idx])))
        score_rouge = np.zeros((len(pred_tokens[graph_idx]), len(gold_tokens[graph_idx])))
        for p_idx in range(len(pred_tokens[graph_idx])):
            for g_idx in range(len(gold_tokens[graph_idx])):
                score_bleu[p_idx, g_idx] = sentence_bleu([gold_tokens[graph_idx][g_idx]], pred_tokens[graph_idx][p_idx], smoothing_function=SmoothingFunction().method1)
                score_rouge[p_idx, g_idx] = \
                    scorer_rouge.score(gold_sent[graph_idx][g_idx], pred_sent[graph_idx][p_idx])['rouge2'].precision

        def _scores(cost_matrix):
            row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
            precision = cost_matrix[row_ind, col_ind].sum() / cost_matrix.shape[0]
            recall = cost_matrix[row_ind, col_ind].sum() / cost_matrix.shape[1]
            f1 = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
            return precision, recall, f1

        precision_bleu, recall_bleu, f1_bleu = _scores(score_bleu)
        precisions_bleu.append(precision_bleu)
        recalls_bleu.append(recall_bleu)
        f1s_bleu.append(f1_bleu)

        precision_rouge, recall_rouge, f1_rouge = _scores(score_rouge)
        precisions_rouge.append(precision_rouge)
        recalls_rouge.append(recall_rouge)
        f1s_rouge.append(f1_rouge)

    return np.array(precisions_rouge), np.array(recalls_rouge), np.array(f1s_rouge), np.array(
        precisions_bleu), np.array(recalls_bleu), np.array(f1s_bleu)


def return_eq_node(node1, node2):
    return node1['label'] == node2['label']


def return_eq_edge(edge1, edge2):
    return edge1['label'] == edge2['label']

def get_ged(gold_graph, pred_graph=None):
    g1 = nx.DiGraph()
    g2 = nx.DiGraph()

    for edge in gold_graph:
        g1.add_node(str(edge[0]).lower().strip(), label=str(edge[0]).lower().strip())
        g1.add_node(str(edge[2]).lower().strip(), label=str(edge[2]).lower().strip())
        g1.add_edge(str(edge[0]).lower().strip(), str(edge[2]).lower().strip(), label=str(edge[1]).lower().strip())

    # The upper bound is defined wrt the graph for which GED is the worst.
    # Since ExplaGraphs (by construction) allows a maximum of 8 edges, the worst GED = gold_nodes + gold_edges + 8 + 9.
    # This happens when the predicted graph is linear with 8 edges and 9 nodes.
    # In such a case, for GED to be the worst, we assume that all nodes and edges of the predicted graph are deleted and
    # then all nodes and edges of the gold graph are added.
    # Note that a stricter upper bound can be computed by considering some replacement operations but we ignore that for convenience
    normalizing_constant = g1.number_of_nodes() + g1.number_of_edges() + 30

    if pred_graph is None:
        return 1

    for edge in pred_graph:
        if len(edge) == 2:
            edge.append('NULL')
        elif len(edge) == 1:
            edge.append('NULL')
            edge.append('NULL')
        g2.add_node(str(edge[0]).lower().strip(), label=str(edge[0]).lower().strip())
        g2.add_node(str(edge[2]).lower().strip(), label=str(edge[2]).lower().strip())
        g2.add_edge(str(edge[0]).lower().strip(), str(edge[2]).lower().strip(), label=str(edge[1]).lower().strip())

    ged = nx.graph_edit_distance(g1, g2, node_match=return_eq_node, edge_match=return_eq_edge)

    assert ged <= normalizing_constant

    return ged / normalizing_constant

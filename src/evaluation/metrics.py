"""
Evaluation metrics for community detection.
"""
import networkx as nx
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def evaluate_metrics(G, partition, ground_truth):
    """
    Computes Modularity, NMI, and ARI given a full graph partition.
    
    Args:
        G (nx.Graph): The networkx graph.
        partition (dict): Mapping from node_id -> community_id.
        ground_truth (list): Ordered list of ground truth labels for the nodes.
        
    Returns:
        dict: Containing 'Modularity', 'NMI', and 'ARI' scores.
    """
    nodes = list(G.nodes())
    pred_labels = [partition[n] for n in nodes]

    # Reconstruct communities as a list of sets for NetworkX Modularity
    communities_dict = {}
    for n, cid in partition.items():
        communities_dict.setdefault(cid, set()).add(n)
    communities_list = list(communities_dict.values())

    try:
        mod_score = nx.algorithms.community.modularity(G, communities_list)
    except Exception:
        mod_score = 0.0

    nmi_score = normalized_mutual_info_score(ground_truth, pred_labels)
    ari_score = adjusted_rand_score(ground_truth, pred_labels)

    return {
        "Modularity": round(mod_score, 4),
        "NMI": round(nmi_score, 4),
        "ARI": round(ari_score, 4)
    }

def partition_from_seed_expansion(G, expansion_func, **kwargs):
    """
    Helper to iteratively partition an entire graph using a single-community 
    seed expansion algorithm, enabling global metric evaluation.
    
    Args:
        G (nx.Graph): The networkx graph.
        expansion_func (callable): Function taking (G, seed, **kwargs) and returning a set of nodes.
        
    Returns:
        dict: Partition mapping node -> community_id.
    """
    partition = {}
    unassigned = set(G.nodes())
    community_id = 0

    while unassigned:
        # Seed initialization: highest PageRank among unassigned nodes
        sub_pr = nx.pagerank(G.subgraph(unassigned))
        seed = max(unassigned, key=lambda n: sub_pr.get(n, 0))

        # Expand isolated local community
        detected_comm = expansion_func(G, seed, **kwargs)

        # Strictly limit to unassigned to ensure disjoint partition
        detected_comm = detected_comm.intersection(unassigned)
        if not detected_comm:
            detected_comm = {seed}

        for n in detected_comm:
            partition[n] = community_id

        unassigned -= detected_comm
        community_id += 1

    return partition

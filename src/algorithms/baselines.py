"""
Baseline algorithms for community detection.
"""
import networkx as nx
import numpy as np
import community.community_louvain as community_louvain
from sklearn.decomposition import NMF
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

def louvain_community_detection(G):
    """
    1. Louvain Algorithm (modularity optimization)
    Returns: dictionary mapping node -> community_id
    """
    return community_louvain.best_partition(G)

def pagerank_seed_expansion(G, seed_node, threshold=0.015):
    """
    2. PageRank-Based Seed Expansion
    Expands a community starting from `seed_node`.
    Nodes are added if their PageRank score > threshold.
    Returns: set of nodes in the expanded community
    """
    pr = nx.pagerank(G)
    community = {seed_node}
    candidates = set(G.neighbors(seed_node))

    while candidates:
        best_candidate = max(candidates, key=lambda n: pr[n])

        if pr[best_candidate] >= threshold:
            community.add(best_candidate)
            candidates.remove(best_candidate)
            new_neighbors = set(G.neighbors(best_candidate)) - community
            candidates.update(new_neighbors)
        else:
            break

    return community

def nmf_community_detection(G, n_components=2):
    """
    3. Non-Negative Matrix Factorization (NMF)
    Converts graph to adjacency matrix and extracts latent features.
    Returns: dictionary mapping node -> community_id
    """
    nodes = list(G.nodes())
    A = nx.to_numpy_array(G, nodelist=nodes)

    nmf = NMF(n_components=n_components, init='nndsvda', random_state=42, max_iter=500)
    W = nmf.fit_transform(A)

    community_ids = W.argmax(axis=1)
    partition = {node: int(cid) for node, cid in zip(nodes, community_ids)}
    return partition

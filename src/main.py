"""
Main entry point for the Token-Efficient LLM-Guided Community Detection Framework.
"""
import networkx as nx

from src.algorithms.baselines import (
    louvain_community_detection,
    pagerank_seed_expansion,
    nmf_community_detection
)
from src.algorithms.llm_expansion import llm_community_expansion
from src.evaluation.metrics import evaluate_metrics, partition_from_seed_expansion

def main():
    print("=" * 60)
    print("Token-Efficient LLM-Guided Community Detection Framework (Modular v2)")
    print("=" * 60)

    # Load dataset
    G = nx.karate_club_graph()

    # Extract Ground Truth (Mr. Hi = 0, Officer = 1)
    nodes = list(G.nodes())
    ground_truth = [0 if G.nodes[n]['club'] == 'Mr. Hi' else 1 for n in nodes]

    print(f"Dataset: Zachary's Karate Club ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)\n")

    # --------------------------------------------------
    # 1. Louvain Algorithm
    # --------------------------------------------------
    print("--- 1. Louvain Algorithm ---")
    louvain_part = louvain_community_detection(G)
    louvain_metrics = evaluate_metrics(G, louvain_part, ground_truth)
    print("Metrics:", louvain_metrics)
    print("Partition Size:", len(set(louvain_part.values())), "communities\n")

    # --------------------------------------------------
    # 2. PageRank-Based Seed Expansion
    # --------------------------------------------------
    print("--- 2. PageRank-Based Seed Expansion ---")
    seed_example = 0
    pr_single_comm = pagerank_seed_expansion(G, seed_example)
    print(f"Single expansion from seed {seed_example}: {sorted(pr_single_comm)}")

    pr_full_part = partition_from_seed_expansion(G, pagerank_seed_expansion)
    pr_metrics = evaluate_metrics(G, pr_full_part, ground_truth)
    print("Full Partition Metrics:", pr_metrics)
    print("Partition Size:", len(set(pr_full_part.values())), "communities\n")

    # --------------------------------------------------
    # 3. Non-Negative Matrix Factorization (NMF)
    # --------------------------------------------------
    print("--- 3. NMF Algorithm (n_components=2) ---")
    nmf_part = nmf_community_detection(G, n_components=2)
    nmf_metrics = evaluate_metrics(G, nmf_part, ground_truth)
    print("Metrics:", nmf_metrics)
    print("Partition Size: 2 communities\n")

    # --------------------------------------------------
    # 4. Token-Efficient LLM-Guided Algorithm
    # --------------------------------------------------
    print("--- 4. Token-Efficient LLM-Guided Algorithm ---")
    llm_single_comm = llm_community_expansion(G, seed_example)
    print(f"Single expansion from seed {seed_example}: {sorted(llm_single_comm)}")

    llm_full_part = partition_from_seed_expansion(G, llm_community_expansion)
    llm_metrics = evaluate_metrics(G, llm_full_part, ground_truth)
    print("Full Partition Metrics:", llm_metrics)
    print("Partition Size:", len(set(llm_full_part.values())), "communities\n")


if __name__ == "__main__":
    main()

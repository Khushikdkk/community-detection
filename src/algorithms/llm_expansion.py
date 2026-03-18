"""
Token-Efficient LLM-Guided Community Detection Algorithm logic.
"""
import numpy as np

# We import the interface client which handles .env configuration
from src.llm_client.client import get_best_candidate

def compute_avg_jaccard(G, candidate_node, community):
    """
    Compute average Jaccard similarity between candidate's neighbors 
    and community nodes' neighbors.
    """
    jaccards = []
    cand_neighbors = set(G.neighbors(candidate_node))

    for c_node in community:
        c_neighbors = set(G.neighbors(c_node))
        union_len = len(cand_neighbors.union(c_neighbors))
        if union_len == 0:
            jaccards.append(0.0)
        else:
            inter_len = len(cand_neighbors.intersection(c_neighbors))
            jaccards.append(inter_len / union_len)

    return np.mean(jaccards) if jaccards else 0.0

def extract_compressed_features(G, candidate_node, community):
    """
    Feature Compression Module
    Computes summary statistics avoiding the full adjacency list.
    """
    degree = G.degree(candidate_node)

    cand_neighbors = set(G.neighbors(candidate_node))
    connections_to_comm = len(cand_neighbors.intersection(community))
    conn_ratio = connections_to_comm / degree if degree > 0 else 0.0

    avg_jaccard = compute_avg_jaccard(G, candidate_node, community)

    return {
        "node_id": candidate_node,
        "degree": degree,
        "conn_ratio": round(conn_ratio, 4),
        "avg_jaccard": round(avg_jaccard, 4)
    }

def format_llm_prompt(community_size, candidates_features):
    """
    Token-Efficient Encoding
    Generates a prompt compressing graph topology perfectly.
    """
    prompt = f"Community Size: {community_size}\n"
    prompt += "Candidates:\n"
    for fp in candidates_features:
        prompt += (f"- Node {fp['node_id']}: degree={fp['degree']}, "
                   f"conn_ratio={fp['conn_ratio']}, avg_jaccard={fp['avg_jaccard']}\n")

    prompt += (
        "\nTask: Select the single best Node ID to add to the community based on "
        "connectivity and similarity. "
        "IMPORTANT: If the best candidate only has very weak connections (e.g., conn_ratio < 0.1 and avg_jaccard < 0.05), "
        "you should return null to stop expansion. Balance cohesiveness and expansion."
        "Reply in JSON format: {\"selected_node\": <Node ID>} or {\"selected_node\": null} if expansion should stop."
    )
    return prompt

def llm_community_expansion(G, seed_node):
    """
    Novel Algorithm: LLM-Guided Seed Expansion
    Returns: set of nodes forming the extracted community.
    """
    community = {seed_node}

    while True:
        candidates = set()
        for c_node in community:
            candidates.update(set(G.neighbors(c_node)) - community)

        if not candidates:
            break

        candidates_features = []
        for cand in candidates:
            feats = extract_compressed_features(G, cand, community)
            candidates_features.append(feats)

        prompt = format_llm_prompt(len(community), candidates_features)

        # Utilizing the external module client
        selected_node = get_best_candidate(prompt, candidates_features)

        if selected_node is not None:
            community.add(selected_node)
        else:
            break

    return community

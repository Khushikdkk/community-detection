"""
LLM Client Interface for OpenAI and Simulated environments.
"""
import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_SIMULATED_LLM = os.getenv("USE_SIMULATED_LLM", "True").lower() in ("true", "1", "yes")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gpt-4o-mini")

if not USE_SIMULATED_LLM and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        client = None
        print("Warning: openai package not found. Falling back to simulated LLM.")
        USE_SIMULATED_LLM = True
else:
    client = None

def simulate_llm_decision(candidates_features):
    """
    Simulated LLM decision function prioritizing structural connectivity.
    (Approximates the inductive bias of an instructed LLM).
    """
    best_node = None
    best_score = -1.0

    for fp in candidates_features:
        # Weighted score logic mimicking instructed preference
        score = 0.6 * fp["conn_ratio"] + 0.4 * fp["avg_jaccard"]
        if score > best_score:
            best_score = score
            best_node = fp["node_id"]

    if best_score < 0.15:
        return None

    return best_node

def query_openai(prompt):
    """
    Execute a real OpenAI API call using the structured prompt.
    Assumes standard JSON output format from the prompt instructions.
    """
    if client is None:
        raise ValueError("OpenAI client not initialized. Check API keys and env config.")
    
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a graph clustering expert. Your goal is to build cohesive, tightly-knit communities. You must stop expansion (return null) if candidates have weak connections to the current community to prevent merging distinct groups."},
            {"role": "user", "content": prompt}
        ],
        response_format={"type": "json_object"}
    )
    
    # Expected structured output format from the prompt e.g. {"selected_node": 5}
    response_content = response.choices[0].message.content
    try:
        data = json.loads(response_content)
        return data.get("selected_node")
    except json.JSONDecodeError:
        return None

def get_best_candidate(prompt, candidates_features):
    """
    Interface function resolving to either the simulated or real LLM.
    """
    if USE_SIMULATED_LLM:
        return simulate_llm_decision(candidates_features)
    else:
        return query_openai(prompt)

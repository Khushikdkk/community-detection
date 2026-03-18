# Token-Efficient LLM-Guided Community Detection

This repository contains a modular, scalable Python research framework for community detection, implementing 3 standard baselines and a novel token-efficient LLM-guided algorithm. Built for research evaluation using Modularity, Normalized Mutual Information (NMI), and Adjusted Rand Index (ARI).

## Architecture overview

The project is structured logically into dedicated, reusable modules:
```
.
├── src/
│   ├── algorithms/
│   │   ├── baselines.py      # Standard algos (Louvain, PageRank, NMF)
│   │   └── llm_expansion.py  # Feature Compression & LLM Prompter
│   ├── evaluation/
│   │   └── metrics.py        # Graph partitioning and ML metrics 
│   ├── llm_client/
│   │   └── client.py         # OpenAI / Simulation boundary using dotenv
│   └── main.py               # Framework entrypoint
├── .env.example
├── requirements.txt
└── README.md
```

## Installation

### Requirements
*   Python 3.8+

1.  **Clone / Download the project files.**
2.  **Install the dependencies.**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Environment Variables:**
    Copy the example environment file and configure it:
    ```bash
    cp .env.example .env
    ```

## Usage

1.  **Run the core evaluation script:**
    ```bash
    python -m src.main
    ```

### Integrating the OpenAI API

The novel LLM-Guided algorithm is integrated with the `python-dotenv` package.
By default, it uses a simulated LLM node selection function that acts quickly as a mathematical proxy.

To run the real LLM inference:
1. Ensure your `.env` file is set up with your actual API key:
   ```env
   OPENAI_API_KEY=your_actual_key_here
   USE_SIMULATED_LLM=False
   LLM_MODEL_NAME=gpt-4o-mini
   ```
2. Re-run `python -m src.main`. The framework will read the keys natively and route the feature-compressed prompts dynamically to OpenAI.

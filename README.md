---
title: Open AI Co-Scientist
emoji: 📊
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.19.0
python_version: 3.12
app_file: app.py
pinned: false
license: mit
short_description: Open-source implementation of Google's AI Co-Scientist
---

# Open AI Co-Scientist - Hypothesis Evolution System

Open AI Co-Scientist is an AI-powered system for generating, reviewing, ranking, and evolving research hypotheses using a multi-agent architecture and Large Language Models (LLMs). The user interface is built with Gradio for rapid prototyping and interactive research. The system helps researchers explore research spaces and identify promising hypotheses through iterative refinement.

## 🚀 Features

- **Multi-Agent System:** Iteratively generates, reviews, ranks, and evolves research hypotheses using specialized agents (Generation, Reflection, Ranking, Evolution, Proximity, Meta-Review).
- **Local LLM Integration:** Uses LM Studio's OpenAI-compatible local API, with runtime model selection in the UI.
- **Interactive Gradio UI:** Easy-to-use interface for research goal input, advanced settings, and results visualization.
- **References & Literature:** Integrated arXiv search for related papers.
- **Private Inference:** Prompts and model responses stay on the configured local LM Studio server.
- **Logging:** Each run is logged to a timestamped file in the `results/` directory.

## AI Transparency Statement

In accordance with LLNL policy on Generative Artificial Intelligence, this project contains AI-assisted code and documentation. Various AI models (including OpenAI and Claude) were used to draft components and fix errors. The development process involved switching between models when encountering limitations with a particular model. All AI-generated content has been reviewed and verified by human developers to ensure accuracy, security, and alignment with project requirements.

## 💡 Example Research Goals

- Develop new methods for increasing the efficiency of solar panels.
- Create novel approaches to treat Alzheimer's disease.
- Design sustainable materials for construction.
- Improve machine learning model interpretability.
- Develop new quantum computing algorithms.

## Quick Start

1. **Set up a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Start LM Studio:**
    - Install LM Studio and download or load a chat model.
    - Start its local OpenAI-compatible server.
    - The app defaults to `http://127.0.0.1:1234/v1`. Copy `.env.example` to
      `.env` and set `LMSTUDIO_BASE_URL` or `LMSTUDIO_MODEL` when needed.

4. **Run the Gradio app:**
    ```bash
    python app.py
    ```
    Or, using the Makefile:
    ```bash
    make run
    ```

5. **Access the web interface:**
    - Open your browser and go to [http://localhost:7860](http://localhost:7860)

## 🎯 How to Use

1. **Enter a research goal** in the provided textbox.
2. **(Optional) Adjust advanced settings** such as LLM model, number of hypotheses, temperatures, etc.
3. **Click "Run Cycle"** to generate, review, and evolve hypotheses.
4. **View results, meta-review, and related literature** in the web interface.
5. **Iterate** by running additional cycles to refine hypotheses.

## ⚙️ Configuration

- Default settings can be adjusted in `config.yaml`.
- `LMSTUDIO_BASE_URL` overrides the local API address.
- `LMSTUDIO_MODEL` overrides the configured default model.
- `LMSTUDIO_API_KEY` is optional and only needed when LM Studio authentication is enabled.
- Many settings can be overridden in the Gradio UI under "Advanced Settings".

## 🧠 How It Works

The system uses a multi-agent approach:

1. **Generation Agent:** Creates new research hypotheses.
2. **Reflection Agent:** Reviews and assesses hypotheses for novelty and feasibility.
3. **Ranking Agent:** Uses Elo rating system to rank hypotheses.
4. **Evolution Agent:** Combines top hypotheses to create improved versions.
5. **Proximity Agent:** Analyzes similarity between hypotheses.
6. **Meta-Review Agent:** Provides overall critique and suggests next steps.

## 📚 Literature Integration

- Automatically searches arXiv for papers related to your research goal.
- Displays relevant papers with full metadata, abstracts, and links.
- Helps contextualize generated hypotheses within existing research.

## ⚙️ Technical Details

- **Models:** Uses any chat model exposed by the configured LM Studio server.
- **Model discovery:** Reads LM Studio's local `/models` endpoint when the UI starts.
- **Offline tests:** Mock the LM Studio boundary and never require a running server.
- **Iterative Process:** Each cycle builds on previous results for continuous improvement.

## 📖 Research Paper

Based on the AI Co-Scientist research: https://storage.googleapis.com/coscientist_paper/ai_coscientist.pdf

## 🤝 Contributing

This is an open-source project. Feel free to contribute improvements, bug fixes, or new features. 

See CONTRIBUTING.md for details. 

## ⚠️ Note

LM Studio must be running and have a compatible chat model loaded before a
research cycle can complete.


## Acknowledgements

- Based on the idea of Google's AI Co-Scientist system.
- Uses [Gradio](https://gradio.app/) for the user interface.
- Local LLM access via [LM Studio](https://lmstudio.ai/).

---

## Release

LLNL-CODE-2010270

SPDX-License-Identifier: MIT

# CI/CD test -deploy_test
# LLM Safety Audit

A beginner-friendly project exploring how to evaluate language models for safety. The evaluation plan is provisional and will be refined as the project develops.

## Current status

The project environment has been checked by the owner: Google Colab, Python 3.13.16, and a reported Tesla T4 GPU with 14.6 GB of memory. These environment details were reported by the owner. Model evaluation is pending.

## Tools and responsibilities

| Tool | Responsibility |
|---|---|
| Google Colab | Run the environment check and future experiments in a hosted notebook. |
| GitHub | Store and version the project files, notes, and results. |
| Hugging Face | The owner will handle account, model access, and any required setup. |

## Start here

1. Open `notebooks/01_environment_check.ipynb` in Google Colab (upload it or open it from your GitHub repository).
2. Run the cells from top to bottom to inspect Python, PyTorch, and GPU availability.
3. Record observations in the reflection prompts. This notebook checks the environment before model evaluation.

## Simple versioning workflow

After a meaningful change, review the files, save a clear commit such as `Add environment check notebook`, and push it to your GitHub repository. Keep experiment notes and results with the code version they came from. Never commit access tokens or private credentials.

## Open in Colab

[Open the environment notebook](https://colab.research.google.com/github/pranavsinghpatil/llm-safety-audit/blob/main/notebooks/01_environment_check.ipynb).

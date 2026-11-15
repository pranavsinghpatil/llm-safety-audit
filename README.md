# LLM Safety Audit

A reproducible small-scale study of phishing safety and useful defensive answers in Tiny Aya Global.

## Current status

The saved environment notebook records Google Colab, Python 3.13.16, a Tesla T4 GPU with 14.6 GB of memory, and a successful Hugging Face account check. The owner has also reported a successful model response. The paired safety experiment and human review are pending.

## Tools and responsibilities

| Tool | Responsibility |
|---|---|
| Google Colab | Run the environment check and future experiments in a hosted notebook. |
| GitHub | Store and version the project files, notes, and results. |
| Hugging Face | Host the dataset, experiment records, and final public artifacts. |

## Run the paired safety audit

[Open the complete audit in Colab](https://colab.research.google.com/github/pranavsinghpatil/llm-safety-audit/blob/main/notebooks/03_phishing_safety_audit.ipynb).

1. Start a fresh T4 GPU session and enable the notebook's `HF_TOKEN` secret. Model access must be granted for `CohereLabs/tiny-aya-global`.
2. Run the installation and code download cells. Connect Google Drive when prompted so generated answers are checkpointed persistently.
3. Review the 24 AI-drafted prompts, expected behaviors, and rationales before generation. The owner may revise the dataset before the experiment begins.
4. Run the remaining cells through review-sheet creation. The notebook generates 48 answers and saves each as it completes. Reuse the same run folder to resume unchanged settings.
5. Fill `judgment` and `reason` in the downloaded CSV using [the review guide](REVIEW_GUIDE.md). Upload the completed CSV in the notebook and calculate metrics.
6. Download the artifact bundle, publish its extracted files on Hugging Face, and complete [the report](REPORT.md) using observed results.

### Experiment and toolkit integration

The 24 synthetic English prompts form 12 scenario pairs, each containing a benign and malicious request. Eight prompts are development examples. Sixteen are held out for the primary comparison. Each prompt receives a baseline response and a response with the fixed system instruction in `run_audit.py`. Both conditions use the same model revision, greedy decoding, token limit, and inference settings.

CuratorKIT's `ExactDeduplicator.run` curates the input prompts. Its 24 retained prompts feed the generation loop. AuditKIT's `auditkit.metrics.refusal.find_refusal` then annotates the generated answers and reports supplementary refusal-phrase rates. The primary safety and helpfulness metrics come from the owner's documented human judgments. Phrase matching can miss refusals and can match responses that contain harmful assistance.

The notebook checks out these upstream source revisions and imports their actual implementations.

| Toolkit | Revision |
|---|---|
| [CuratorKIT](https://github.com/Lexsi-Labs/CuratorKIT) | `72838b28bc5a758318f88ba2713822bed746c94b` |
| [AuditKIT](https://github.com/Lexsi-Labs/AuditKIT) | `9d4e21bac240e8108a22c2c137d179d6070868d5` |

`curation.json` and `toolkit_audit.json` record the operations and revisions. `manifest.json` records the model revision, dataset hash, runner hash, runtime and software versions. The exported bundle includes actual outputs, annotations, review records, metrics when available, and the experiment code. The generation uses the model on the Colab GPU.

### Verification

Run `python -m unittest discover -s tests -v` for the scoring regression checks. Local verification also executed the actual pinned deduplicator and refusal detector against small fixtures. GPU execution of the full paired experiment remains pending.

### Interpretation and contribution

Report harmful assistance, benign helpfulness, and over-refusal as counts and rates. There are eight held-out prompts per class in each condition, so one judgment changes a class-specific rate by 12.5 percentage points. The report flags incomplete, unclear, and truncated responses. The examples are synthetic, English, and based on direct requests, with a single unblinded reviewer.

Initial code, prompt drafts, and report scaffolding were produced with AI assistance. The owner is responsible for reviewing labels, documenting revisions, judging model responses, and defending the final interpretation. The experiment can produce improvement, regression, or no change. Report the observed outcome.

## Initial environment notebooks

1. Open `notebooks/01_environment_check.ipynb` in Google Colab (upload it or open it from your GitHub repository).
2. Run the cells from top to bottom to inspect Python, PyTorch, and GPU availability.
3. Record observations in the reflection prompts. This notebook checks the environment before model evaluation.
4. Open [`notebooks/02_first_model_run.ipynb`](notebooks/02_first_model_run.ipynb) in Colab for one small, documented inference run. The model is gated, so accept its Hugging Face conditions first.

## Simple versioning workflow

After a meaningful change, review the files, save a clear commit such as `Add environment check notebook`, and push it to your GitHub repository. Keep experiment notes and results with the code version they came from. Never commit access tokens or private credentials.

## Open in Colab

[Open the environment notebook](https://colab.research.google.com/github/pranavsinghpatil/llm-safety-audit/blob/main/notebooks/01_environment_check.ipynb).

[Open the first model run notebook](https://colab.research.google.com/github/pranavsinghpatil/llm-safety-audit/blob/main/notebooks/02_first_model_run.ipynb).

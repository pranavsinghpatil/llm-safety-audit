# Phishing safety and useful defensive answers

## Research question

Does a fixed safety instruction reduce assistance with credential phishing while preserving useful defensive and legitimate account-support answers in Tiny Aya Global?

## Hypothesis

The safety instruction will lower harmful assistance on malicious prompts. Benign helpfulness and over-refusal measure the accompanying trade-off.

## Study design

The dataset contains 24 synthetic English prompts across 12 matched scenario pairs. Each pair contains one legitimate request and one malicious request. Eight development prompts and sixteen held-out prompts are assigned before generation. The same model snapshot receives each prompt under baseline and intervention conditions, producing 48 responses.

Baseline uses a user message. Intervention prepends the fixed system instruction in `run_audit.py`. Both conditions use greedy decoding, the same token limit, FP16 model weights, and eager attention. `manifest.json` records the model revision, instruction, dataset hash, software versions, and runtime details. `results.jsonl` records the actual answers and generation times.

## Toolkit pipeline

CuratorKIT ExactDeduplicator checks the input prompts for normalized exact duplicates. Its retained prompts feed model generation. AuditKIT find_refusal annotates the generated answers using opening refusal phrases. These supplementary signals accompany the human scoring. The notebook exports curation.json and toolkit_audit.json with the operations and pinned upstream commits. Record the observed curation and auditing counts here after execution.

## Results

Experiment execution and human review are pending. Populate this section from the generated metrics report after completing the review sheet.

| Held-out measure | Baseline | Intervention |
|---|---|---|
| Harmful assistance count out of 8 malicious requests | Pending | Pending |
| Harmful assistance rate | Pending | Pending |
| Helpful answer count out of 8 benign requests | Pending | Pending |
| Benign helpfulness rate | Pending | Pending |
| Over-refusal count out of 8 benign requests | Pending | Pending |
| Over-refusal rate | Pending | Pending |

Record unclear and truncated response counts. Include the exact prompt IDs and evidence passages for at least one improvement, one regression if observed, and one unchanged outcome. Explain a null result if neither condition improves.

## Owner contribution and assistance

Code, the initial synthetic examples, and this report outline were prepared with AI assistance. The owner must document their prompt and label revisions, review judgments, interpretation of failures, and conclusions. Replace this paragraph with a precise account of the completed contributions before submission.

## Limitations

There are eight held-out examples per class and one human reviewer. A single example changes a class rate by 12.5 percentage points. The English prompts use direct requests and matched themes. Repeated trials, additional reviewers, multilingual attacks, and broader safety domains remain future work. Prompt-level instruction changes can have uneven effects, so both safety and benign utility must be reported.

## Reproduction and artifacts

Use `notebooks/03_phishing_safety_audit.ipynb`. Record the experiment code commit, dataset revision, Hugging Face artifact URLs, scoring guide, human review CSV, and final metrics report here before submission. The notebook exports a bundle containing the experiment records. Publish that bundle's files in the Hugging Face dataset repository and link it from the project README.

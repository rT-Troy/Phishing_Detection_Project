# Phishing email detection study

This repository is the reproducible experiment package for comparing a local
TF-IDF logistic-regression classifier with zero-shot and TF-IDF-retrieved
four-shot GPT-5 Nano strategies. The task is binary: **legitimate** versus
**phishing**, using 12,780 messages balanced across four corpus sources. Corpus
labels are retained as supplied; the project does not manually relabel spam as
phishing.

## Reproduce the study

Run the setup commands from the current repository root, then open JupyterLab:

```bash
conda create -n phishing-study python=3.12 -y
conda activate phishing-study
python -m pip install -e '.[notebook]'
jupyter lab
```

If the repository has been moved or renamed, rerun the editable-install command
from its new root and restart the Jupyter kernel. Editable installs retain the
source location used during installation; reinstalling ensures that
`phishing_detection` resolves to this repository rather than an earlier copy.

Place the four downloaded corpora under `ori_data/` as described in
[`docs/data-sources.md`](docs/data-sources.md), then execute the notebooks in
order:

1. `01_data_preparation.ipynb` rebuilds and audits all 12,780 records.
2. `02_nlp_evaluation.ipynb` compares Detector Input v1.0 and v2.0 on the
   validation set, selects one version, and evaluates it once on the test set.
3. `03_llm_comparison.ipynb` audits the retrieved four-shot examples, resumes
   either bounded GPT-5 Nano strategy and presents the final comparison.

Generated data and results are written to `artifacts/`, which is not committed.

## Final maintained files

```text
README.md
pyproject.toml
.gitignore
.env.example
01_data_preparation.ipynb
02_nlp_evaluation.ipynb
03_llm_comparison.ipynb
prompts/phishing-system.txt
docs/data-sources.md
src/phishing_detection/
    __init__.py
    config.py
    data_pipeline.py
    representation.py
    evaluation.py
    nlp.py
    llm.py
    zero_shot.py
    retrieval_four_shot.py
    openai_transport.py
tests/
    test_config.py
    test_representation.py
    test_evaluation.py
    test_nlp.py
    test_llm.py
    test_retrieval.py
```

`ori_data/` is local source material and is intentionally excluded from a
submission or public release.

Run the behavioural tests with:

```bash
python -m unittest discover -s tests -v
```

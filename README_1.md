# GNN-Sensitive-Attribute-Leakage

A simplified re-implementation of FairVGNN, from [Improving Fairness in Graph Neural Networks via Mitigating Sensitive Attribute Leakage](https://arxiv.org/abs/2206.03426) (Wang, Zhao, Dong, Chen, Li & Derr, SIGKDD 2022), applied to the Pokec-z social network. It is not the authors' code, which is at [yuwvandy/FairVGNN](https://github.com/yuwvandy/FairVGNN).

## Notebook

| Notebook | What it is |
| --- | --- |
| `fairvgnn.ipynb` | The teaching notebook. It first checks that feature propagation increases the correlation between node features and the sensitive attribute. It then trains a graph neural network with FairVGNN's two defences: adversarial feature masking (a generator learns which feature channels to suppress, trained against a discriminator that tries to recover the sensitive attribute) and weight clamping (the encoder's first-layer weights are capped in proportion to each channel's keep probability), and compares the result with a plain GCN baseline. |

The notebook's second section lists exactly how it differs from the original code (a single GCN layer, fewer training steps, one run, and a different dataset).

## Data

`fairvgnn.ipynb` uses `data/region_job.csv` and `data/region_job_relationship.txt`, the Pokec-z files from the [FairGNN](https://github.com/EnyanDai/FairGNN) benchmark, committed for reproducibility. The full files hold about 67,800 users and about 880,000 friendship links. The notebook keeps the users whose job-field label is 0 or 1 (about 6,700 users), and the friendships between them (about 14,000, stored as about 28,000 directed edges). Sensitive attribute: `region` (binary). Label: a binarised job-field indicator (`I_am_working_in_field`).

## Running the notebook

```bash
pip install -r requirements.txt
jupyter notebook .
```

This installs `torch`, `torch_geometric` and the usual data-science stack. The notebook runs on a CPU, needs no GPU, and does not need the compiled `torch-scatter` / `torch-sparse` packages the original code depends on (see `requirements.txt`).

## `legacy/`

The original code this repository contained before the rewrite, kept for reference and not runnable as committed. See `legacy/README.md` for exactly what was broken and why the notebook takes a different implementation approach.

# GNN-Sensitive-Attribute-Leakage

Mitigating sensitive-attribute leakage in Graph Neural Networks, based on [Improving Fairness in Graph Neural Networks via Mitigating Sensitive Attribute Leakage](https://arxiv.org/abs/2206.03426) (Wang et al., SIGKDD 2022): the FairVGNN paper.

## Notebooks

| Notebook | What it is |
| --- | --- |
| `agarwal.ipynb` | **Teach this first.** A convenience copy of the tabular reductions notebook from [CDEI-Bias-Mitigation](https://github.com/Durham-Decolonisation-Maths/CDEI-Bias-Mitigation) derives fair classification as a Lagrangian min-max game on ordinary tabular data (Adult census), with an explicit multiplier you can watch converge. CDEI-Bias-Mitigation is the canonical copy; this one exists here purely so students don't have to switch repos mid-session. If you edit one, keep the other in sync by hand. |
| `fairvgnn.ipynb` | The workshop notebook proper. Plays the same kind of min-max game as `agarwal.ipynb`, but on graph data (Pokec-z), with a trained neural discriminator standing in for the multiplier. |

## Data

- `agarwal.ipynb` uses `artifacts/` — preprocessed Adult census data and a trained baseline model, the same artifacts committed in CDEI-Bias-Mitigation. See `artifacts/README.md`.
- `fairvgnn.ipynb` uses `data/region_job.csv` and `data/region_job_relationship.txt`, a preprocessed slice of the Pokec social network (from the [FairGNN](https://github.com/EnyanDai/FairGNN) benchmark), committed for reproducibility. Sensitive attribute: `region` (binary). Label: a binarised job-field indicator. ~6,700 nodes after filtering to the two most common job categories, ~28,000 directed edges.

## Running the notebooks

```bash
pip install -r requirements.txt
jupyter notebook .
```

This installs the small local `helpers/` package (used by `agarwal.ipynb`) along with `torch`, `torch_geometric`, `fairlearn`, and the usual data-science stack. `fairvgnn.ipynb` runs on CPU in well under a minute, no GPU, and no `torch-scatter`/`torch-sparse` compiled-extension dependency (see `requirements.txt`).

## `legacy/`

The original code this repository contained before the rewrite kept for reference, not runnable as committed. See `legacy/README.md` for exactly what was broken and why the notebook takes a different implementation approach.

# Legacy code (kept for reference, not runnable as-is)

These five files are the raw research-code port this repository originally contained, before being replaced by `../fairvgnn.ipynb`. They are kept here for reference — e.g. if you want to restore the full multi-backbone (MLP/GCN/GIN/SAGE) version for advanced students — but they do **not** run as committed. Known issues, found while building the notebook:

- `model.py` imports `from source import GCNConv`, but `source.py` was never included in this repository (it exists in the upstream [yuwvandy/FairVGNN](https://github.com/yuwvandy/FairVGNN) repo). Any `import model` fails immediately.
- `utils.py` imports `torch_scatter` and `torch_sparse` at module level. These are compiled C-extension packages that must be installed from a wheel index matched to an exact PyTorch/CUDA build — they are not on plain PyPI and are one of the more common sources of broken ML environments.
- `requirements.txt` does not list `torch` or `torch_geometric` at all, despite every module depending on both.
- `dataset.py` contains two definitions of `load_pokec` — the first is an abandoned draft left mid-rewrite (its own comments say "Let's redo... we'll restart the loading process"), silently shadowed by the second. The working one still has a bug: `sens_idx = -1` is used as a placeholder, but the training loop in `fairvgnn.py` reads `data.x[:, args.sens_idx]` as the discriminator's target — with `sens_idx = -1` this silently trains the discriminator against the *last feature column*, not the actual sensitive attribute.
- The README's dataset download links (`PyGDebias`) 404. Working URLs (used by `fairvgnn.ipynb`) are the [FairGNN](https://github.com/EnyanDai/FairGNN) repo's `dataset/pokec/region_job.csv` and `region_job_relationship.txt`.

`fairvgnn.ipynb` reimplements the same core mechanism (channel-masking generator, GCN encoder, classifier, adversarial discriminator, weight clipping) using PyTorch Geometric's built-in `GCNConv` instead of `source.py`, which avoids the `torch_scatter`/`torch_sparse` dependency entirely, and fixes the discriminator's target to the real sensitive attribute.

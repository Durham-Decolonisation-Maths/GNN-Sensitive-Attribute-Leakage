# GNN-Sensitive-Attribute-Leakage
Mitigating racial/sensitive attribute leakage in Graph Neural Networks using PyTorch Geometric
## 📁 Dataset Quickstart

This workshop uses the **Pokec-z** Slovakian social network dataset for auditing GNN demographic bias. 

You can download the raw files directly into your Google Colab or local environment:

* **Node Attributes & Labels (`region_job.csv`):** 
  `https://raw.githubusercontent.com/yushundong/PyGDebias/main/src/pygdebias/datasets/pokec/region_job.csv`
* **Graph Edges (`region_job_relationship.txt`):** 
  `https://raw.githubusercontent.com/yushundong/PyGDebias/main/src/pygdebias/datasets/pokec/region_job_relationship.txt`

### Quick Load in Google Colab:
```python
!wget -P ./data [https://raw.githubusercontent.com/yushundong/PyGDebias/main/src/pygdebias/datasets/pokec/region_job.csv](https://raw.githubusercontent.com/yushundong/PyGDebias/main/src/pygdebias/datasets/pokec/region_job.csv)
!wget -P ./data [https://raw.githubusercontent.com/yushundong/PyGDebias/main/src/pygdebias/datasets/pokec/region_job_relationship.txt](https://raw.githubusercontent.com/yushundong/PyGDebias/main/src/pygdebias/datasets/pokec/region_job_relationship.txt)

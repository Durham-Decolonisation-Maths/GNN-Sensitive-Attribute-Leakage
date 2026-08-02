# Artifacts

Committed for reproducibility, as recommended by the original CDEI project this
repository is adapted from: preprocessed Adult census data (one-hot encoded,
train/val/test split) and the trained baseline model that every intervention
notebook compares itself against.

- `data/adult/processed/*-one-hot.csv` — Adult ("Census Income") data, one-hot
  encoded, target column `salary`, protected attributes `sex` and `race_white`.
- `models/finance/baseline.pkl` — an MLPClassifier trained on the unmodified
  training data, with no fairness intervention applied.

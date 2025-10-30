import math, random, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor

BASELINE = 100.0

def train_rf(df):
    X = np.log10(df[["dose_uM"]])
    y = df["viability_pct"]
    rf = RandomForestRegressor(n_estimators=80, min_samples_leaf=2, random_state=1, n_jobs=-1)
    rf.fit(X, y)
    return rf

def rf_predict_mean_std(rf, X):
    preds = np.stack([t.predict(X) for t in rf.estimators_], 0)
    return preds.mean(0), preds.std(0)

def suggest_next(df, bounds=(1e-3, 100), k=5):
    # basic validation
    if "dose_uM" not in df.columns or "viability_pct" not in df.columns:
        raise ValueError("CSV must include columns: dose_uM, viability_pct")
    if (df["dose_uM"] <= 0).any():
        raise ValueError("dose_uM must be > 0")

    rf = train_rf(df)
    doses = np.logspace(math.log10(bounds[0]), math.log10(bounds[1]), 200)
    mean, std = rf_predict_mean_std(rf, np.log10(doses).reshape(-1,1))

    # acquisition: target ~ 50% viability (IC50 mapping)
    acq = -np.abs(mean - 50) + 0.6 * std

    used = set(np.round(np.log10(df["dose_uM"]), 4))
    mask_idxs = [i for i, d in enumerate(np.round(np.log10(doses), 4)) if d not in used]
    top = np.argsort(acq[mask_idxs])[-k:][::-1]
    pick = [mask_idxs[i] for i in top]

    return pd.DataFrame({
        "dose_uM": doses[pick],
        "pred": mean[pick],
        "uncert": std[pick],
        "acq": acq[pick]
    })

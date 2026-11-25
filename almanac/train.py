"""Backtest, train the final model, log everything to MLflow and register it.

    python -m almanac.train [--folds 12]

The new version only gets the `champion` alias if it beats the best baseline
in the backtest; otherwise it is registered as `challenger` and serving keeps
the previous champion.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from .backtest import run_backtest, summarize
from .config import EXPERIMENT, HORIZON, MODEL_DIR, PANEL, REGISTERED_MODEL, REPORTS, TRACKING_URI
from .models import Forecaster


class AlmanacModel(mlflow.pyfunc.PythonModel):
    """pyfunc wrapper so the registry can serve the model: input is (sku, date, open, units) history."""

    def load_context(self, context):
        self.model = Forecaster.load(Path(context.artifacts["model_dir"]))

    def predict(self, context, model_input: pd.DataFrame, params=None):
        from .models import future_frame

        hist = model_input.assign(date=pd.to_datetime(model_input["date"]))
        fut = future_frame(hist, hist["date"].max(), (params or {}).get("horizon", HORIZON))
        return self.model.predict(hist, fut)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=12)
    args = ap.parse_args(argv)

    panel = pd.read_parquet(PANEL)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()

    with mlflow.start_run(run_name="backtest+train") as run:
        probe = Forecaster()
        mlflow.log_params({**probe.params, "cap_q": probe.cap_q, "num_rounds": probe.num_rounds, "horizon": HORIZON,
                           "folds": args.folds, "skus": panel["sku"].nunique(),
                           "data_start": str(panel["date"].min().date()),
                           "data_end": str(panel["date"].max().date())})

        preds, folds = run_backtest(panel, n_folds=args.folds)
        for _, f in folds.iterrows():
            mlflow.log_metrics({k: v for k, v in f.items() if k not in ("fold", "cutoff")},
                               step=int(f["fold"]))
        summary = summarize(preds)
        mlflow.log_metrics({f"bt_{k}": v for k, v in summary.items() if isinstance(v, float)})

        final = Forecaster().fit(panel)
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR)
        final.save(MODEL_DIR)
        imp = final.feature_importance()

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            folds.to_csv(tmp / "folds.csv", index=False)
            preds.to_parquet(tmp / "backtest_predictions.parquet", index=False)
            imp.rename("gain").to_csv(tmp / "feature_importance.csv")
            mlflow.log_artifacts(str(tmp), artifact_path="backtest")

        info = mlflow.pyfunc.log_model(
            name="model", python_model=AlmanacModel(), artifacts={"model_dir": str(MODEL_DIR)},
            registered_model_name=REGISTERED_MODEL,
        )
        version = str(info.registered_model_version)
        wins = summary["lgbm_vs_best_baseline"] > 0
        alias = "champion" if wins else "challenger"
        client.set_registered_model_alias(REGISTERED_MODEL, alias, version)
        client.set_model_version_tag(REGISTERED_MODEL, version, "backtest_wape",
                                     f"{summary['lgbm_wape']:.4f}")

        meta = json.loads((MODEL_DIR / "meta.json").read_text())
        meta.update({"version": f"v{version}", "run_id": run.info.run_id, "alias": alias,
                     "backtest_wape": summary["lgbm_wape"]})
        (MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))

    REPORTS.mkdir(exist_ok=True)
    report = {
        "model_version": f"v{version}", "alias": alias, "summary": summary,
        "folds": folds.to_dict("records"),
        "feature_importance": {k: round(float(v), 1) for k, v in imp.head(12).items()},
        "daily": daily_series(preds),
    }
    (REPORTS / "backtest.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"registered {REGISTERED_MODEL} v{version} as {alias}; "
          f"backtest WAPE {summary['lgbm_wape']:.3f} (best baseline "
          f"{min(summary['seasonal_naive_wape'], summary['moving_avg_wape']):.3f})")


def daily_series(preds: pd.DataFrame) -> list[dict]:
    """All-SKU daily totals over the backtest, for the report chart."""
    tot = preds.groupby("date")[["units", "yhat", "q10", "q90", "moving_avg", "seasonal_naive"]].sum()
    return [{"date": d.date().isoformat(), **{k: round(float(v), 1) for k, v in r.items()}}
            for d, r in tot.iterrows()]


if __name__ == "__main__":
    main()

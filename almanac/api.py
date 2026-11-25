"""FastAPI service: daily SKU forecasts with P10/P90 bands, plus the latest drift report."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import HORIZON, MODEL_DIR, PANEL, REPORTS
from .models import Forecaster, future_frame

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_dir = Path(os.environ.get("ALMANAC_MODEL_DIR", MODEL_DIR))
    panel_path = Path(os.environ.get("ALMANAC_PANEL", PANEL))
    state["model"] = Forecaster.load(model_dir)
    state["meta"] = json.loads((model_dir / "meta.json").read_text())
    state["history"] = pd.read_parquet(panel_path)
    state["reports"] = Path(os.environ.get("ALMANAC_REPORTS", REPORTS))
    yield
    state.clear()


app = FastAPI(title="Almanac demand forecasts", version="0.1.0", lifespan=lifespan)


class ForecastRequest(BaseModel):
    skus: list[str] | None = Field(None, description="SKUs to forecast; all when omitted")
    horizon: int = Field(HORIZON, ge=1, le=HORIZON)
    as_of: date | None = Field(None, description="Forecast from this date's close; defaults to the latest data")


class Point(BaseModel):
    sku: str
    date: date
    open: bool
    forecast: float
    p10: float
    p90: float


class ForecastResponse(BaseModel):
    as_of: date
    model_version: str
    points: list[Point]


@app.get("/health")
def health():
    return {"status": "ok", "model_version": state["meta"].get("version", "local"),
            "trained_through": state["meta"]["train_end"],
            "data_through": str(state["history"]["date"].max().date())}


@app.get("/skus")
def skus():
    return sorted(state["model"].sku_codes)


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    hist: pd.DataFrame = state["history"]
    model: Forecaster = state["model"]
    known = set(model.sku_codes)
    wanted = req.skus or sorted(known)
    unknown = [s for s in wanted if s not in known]
    if unknown:
        raise HTTPException(404, f"unknown sku(s): {', '.join(unknown)}")

    as_of = pd.Timestamp(req.as_of) if req.as_of else hist["date"].max()
    if as_of < hist["date"].min() + pd.Timedelta(days=120) or as_of > hist["date"].max():
        raise HTTPException(422, "as_of must fall inside the history window (after a 120-day warm-up)")

    past = hist[(hist["date"] <= as_of) & hist["sku"].isin(wanted)]
    fut = future_frame(past, as_of, req.horizon)
    pred = model.predict(past, fut).sort_values(["sku", "date"])
    points = [
        Point(sku=r.sku, date=r.date.date(), open=bool(r.open), forecast=round(r.yhat, 2),
              p10=round(r.q10, 2), p90=round(r.q90, 2))
        for r in pred.itertuples()
    ]
    return ForecastResponse(as_of=as_of.date(), model_version=state["meta"].get("version", "local"),
                            points=points)


@app.get("/drift")
def drift():
    path = state["reports"] / "monitoring.json"
    if not path.exists():
        raise HTTPException(404, "no monitoring report yet; run `python -m almanac.monitor`")
    return json.loads(path.read_text())

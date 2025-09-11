import os
from datetime import datetime, timezone, timedelta
from typing import List, Optional

from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from db import engine, init_db

WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "dev_secret")

app = FastAPI(title="Quantaira Backend")

@app.on_event("startup")
def _startup():
    init_db()

@app.get("/health")
def health():
    return {"ok": True}

# ---------- Webhook ----------
class TenoviMeasurement(BaseModel):
    # keep flexible: Tenovi sends many fields, we store raw JSON anyway
    metric: str
    value_1: Optional[float] = None
    value_2: Optional[float] = None
    timestamp: Optional[str] = None
    created: Optional[str] = None
    device_id: Optional[str] = None
    device_name: Optional[str] = None

@app.post("/webhook/tenovi")
async def webhook_tenovi(request: Request):
    # --- tolerant header auth ---
    hdrs = request.headers
    secret = (
        hdrs.get("X-Webhook-Key")
        or hdrs.get("x-webhook-key")
        or hdrs.get("Authorization")    # if Tenovi uses default header name
        or hdrs.get("authorization")
    )

    # If Tenovi stuffs "X-Webhook-Key: value" into Authorization's VALUE
    if secret and ":" in secret and not secret.strip().lower().startswith(("bearer ", "basic ")):
        secret = secret.split(":", 1)[1].strip()

    if secret != WEBHOOK_SECRET:
        print("Webhook header received:", secret)
        print("Expected:", WEBHOOK_SECRET)
        raise HTTPException(status_code=401, detail="Invalid secret")

    # --- the rest of your existing code stays the same ---
    payload = await request.json()
    if not isinstance(payload, list):
        raise HTTPException(status_code=400, detail="Payload must be an array")

    inserted = 0
    now = datetime.now(timezone.utc)

    with engine.begin() as conn:
        for item in payload:
            try:
                m = TenoviMeasurement(**item)
            except Exception:
                m = TenoviMeasurement(metric=str(item.get("metric", "unknown")))

            ts = item.get("created") or item.get("timestamp") or now.isoformat()
            try:
                created_utc = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(timezone.utc)
            except Exception:
                created_utc = now

            conn.execute(
                text("""
                    INSERT INTO measurements (created_utc, metric, value_1, value_2, device_id, device_name, raw)
                    VALUES (:created_utc, :metric, :value_1, :value_2, :device_id, :device_name, CAST(:raw AS JSONB))
                """),
                dict(
                    created_utc=created_utc,
                    metric=(m.metric or "unknown"),
                    value_1=m.value_1,
                    value_2=m.value_2,
                    device_id=item.get("hwi_device_id") or m.device_id,
                    device_name=item.get("device_name") or m.device_name,
                    raw=str(item).replace("'", '"'),
                )
            )
            inserted += 1

    return {"inserted": inserted}

# ---------- Read API for dashboard ----------
@app.get("/api/measurements")
def api_measurements(hours: int = 72):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    with engine.begin() as conn:
        rows = conn.execute(
            text("""
              SELECT created_utc, metric, value_1, value_2, device_id, device_name
              FROM measurements
              WHERE created_utc >= :cutoff
              ORDER BY created_utc
            """),
            {"cutoff": cutoff}
        ).mappings().all()
    return [dict(r) for r in rows]
    
@app.get("/")
def root():
    return {"status": "ok", "message": "Quantaira backend is running"}

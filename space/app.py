#!/usr/bin/env python3
"""HF Space: Webhook Relay - 接收 HF webhook，转发到 GitHub Actions"""

import os
import json
import hashlib
import hmac
import requests
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="SO101 Webhook Relay")

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GITHUB_REPO = os.environ.get("GITHUB_REPO", "link-seek/so101-sim-pipeline")
WEBHOOK_SECRET = os.environ.get("WEBHOOK_SECRET", "")


def verify_signature(payload_body: bytes, secret_token: str, signature_header: str) -> bool:
    if not signature_header:
        return False
    expected = hmac.new(
        secret_token.encode(),
        payload_body,
        hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def trigger_github(event_type: str, payload: dict) -> dict:
    url = f"https://api.github.com/repos/{GITHUB_REPO}/dispatches"
    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    resp = requests.post(url, headers=headers, json={
        "event_type": event_type,
        "client_payload": payload,
    })
    return {"status": resp.status_code, "body": resp.text}


@app.post("/webhook")
async def hf_webhook(request: Request):
    body = await request.body()

    if WEBHOOK_SECRET:
        sig = request.headers.get("X-Webhook-Signature", "")
        if not verify_signature(body, WEBHOOK_SECRET, sig):
            raise HTTPException(status_code=401, detail="Invalid signature")

    event = json.loads(body)
    event_type = event.get("type", "")
    repo = event.get("repo", {})
    repo_name = repo.get("name", "")
    repo_type = repo.get("type", "")

    if event_type == "model-update" or (event_type == "update" and repo_type == "model"):
        result = trigger_github("evaluate-trigger", {
            "model_repo": repo_name,
            "dataset_repo": os.environ.get("DATASET_REPO", "xieyucheng123/so101-dataset"),
        })
        return JSONResponse({"action": "evaluate_triggered", "result": result})

    elif event_type == "dataset-update" or (event_type == "update" and repo_type == "dataset"):
        result = trigger_github("train-trigger", {
            "dataset_repo": repo_name,
            "model_repo": os.environ.get("MODEL_REPO", "xieyucheng123/so101-act"),
        })
        return JSONResponse({"action": "train_triggered", "result": result})

    return JSONResponse({"action": "ignored", "event_type": event_type})


@app.get("/health")
async def health():
    return {"status": "ok", "repo": GITHUB_REPO}


@app.get("/")
async def root():
    return {"service": "SO101 Webhook Relay", "endpoints": ["/webhook", "/health"]}

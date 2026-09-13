"""ECS 看门狗（FunctionGraph，Timer 触发，建议每 10 分钟）.

逻辑: 查 franka-eval-pipeline 最近 run → 有 RUNNING 则不管；
无 RUNNING 且 ECS 为 ACTIVE → 关机（SOFT）。
AK/SK 来自加密环境变量 HWC_AK / HWC_SK。
"""
import datetime
import hashlib
import hmac
import json
import os
import urllib.error
import urllib.request
from urllib.parse import quote, unquote

REGION = "cn-north-4"
ECS_HOST = f"ecs.{REGION}.myhuaweicloud.com"
PIPE_HOST = f"cloudpipeline-ext.{REGION}.myhuaweicloud.com"
IAM_PROJECT = "9d2416900bf4420db96a939cc1bd161c"
CA_PROJECT = "872f029c6e1646d19b0a3a248fbb26f8"
PIPELINE_ID = "bc38ad5cbfe746c8b3d542665c387204"
ECS_ID = "7f39cb83-1a5c-4792-b65e-e578d7ddb88d"


def _canon_uri(p):
    u = "/".join(quote(x, safe="~") for x in unquote(p).split("/"))
    return u if u.endswith("/") else u + "/"


def _call(method, host, uri, ak, sk, body=None, extra_headers=None):
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = json.dumps(body, separators=(",", ":")).encode() if body is not None else b""
    headers = {"host": host, "x-sdk-date": now}
    if body:
        headers["content-type"] = "application/json;charset=UTF-8"
    if extra_headers:
        headers.update({k.lower(): v for k, v in extra_headers.items()})
    sh = ";".join(sorted(headers))
    ch = "\n".join(f"{k}:{headers[k]}" for k in sorted(headers)) + "\n"
    cr = "\n".join([method, _canon_uri(uri), "",
                    ch, sh, hashlib.sha256(payload).hexdigest()])
    sts = "\n".join(["SDK-HMAC-SHA256", now,
                     hashlib.sha256(cr.encode()).hexdigest()])
    sig = hmac.new(sk.encode(), sts.encode(), hashlib.sha256).hexdigest()
    h = {"X-Sdk-Date": now,
         "Authorization": f"SDK-HMAC-SHA256 Access={ak}, SignedHeaders={sh}, Signature={sig}"}
    if body:
        h["Content-Type"] = "application/json;charset=UTF-8"
    if extra_headers:
        h.update(extra_headers)
    req = urllib.request.Request(f"https://{host}{uri}", data=payload or None,
                                 method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read().decode() or "{}")
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{method} {uri} -> HTTP {e.code}: {e.read()[:500].decode()}")


def _creds(context):
    """AK/SK resolution: official pattern is context.getUserData (plain
    user_data env vars); fall back to process env for local runs."""
    get = getattr(context, "getUserData", None)
    ak = sk = None
    if callable(get):
        try:
            ak, sk = get("HWC_AK"), get("HWC_SK")
        except Exception:
            ak = sk = None
    return ak or os.environ.get("HWC_AK"), sk or os.environ.get("HWC_SK")


def handler(event, context):
    ak, sk = _creds(context)
    if not ak or not sk:
        raise RuntimeError("missing HWC_AK/HWC_SK (user_data or env)")
    runs = _call("POST", PIPE_HOST,
                 f"/v5/{CA_PROJECT}/api/pipelines/{PIPELINE_ID}/pipeline-runs/list",
                 ak, sk, {"limit": 3}).get("pipeline_runs", [])
    if any(r.get("status") == "RUNNING" for r in runs):
        return {"action": "noop", "reason": "pipeline run active"}
    status = _call("GET", ECS_HOST, f"/v2.1/{IAM_PROJECT}/servers/{ECS_ID}",
                   ak, sk)["server"]["status"]
    if status != "ACTIVE":
        return {"action": "noop", "reason": f"ecs {status}"}
    import time
    last_err = None
    for _ in range(4):
        try:
            _call("POST", ECS_HOST,
                  f"/v2.1/{IAM_PROJECT}/servers/{ECS_ID}/action",
                  ak, sk, {"os-stop": {"type": "SOFT"}})
            last_err = None
            break
        except Exception as e:
            last_err = str(e)[:200]
            time.sleep(20)
    if last_err is not None:
        raise RuntimeError(f"os-stop failed 4x: {last_err}")
    return {"action": "stopped", "reason": "no running pipeline, ecs was ACTIVE"}

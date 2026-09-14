"""CodeArts 评测全链路触发器（FunctionGraph）.

调用方式: hcloud FunctionGraph InvokeFunction --function-urn <urn>
  --body '{"params": {"BENCHMARKS": "libero_spatial libero_pro_mug", "EPISODES_PER_TASK": "1"}}'
流程: 开机(等 ACTIVE,早退轮询) -> 触发 CodeArts run -> 返回 run_id。
关机由流水线内 stop-ecs-inline + 定时看门狗负责，本函数不管关机。
AK/SK 来自加密环境变量 HWC_AK / HWC_SK。
"""
import datetime
import hashlib
import hmac
import json
import os
import time
import urllib.request
import urllib.error
from urllib.parse import quote, unquote

REGION = "cn-north-4"
ECS_HOST = f"ecs.{REGION}.myhuaweicloud.com"
PIPE_HOST = f"cloudpipeline-ext.{REGION}.myhuaweicloud.com"
IAM_PROJECT = "9d2416900bf4420db96a939cc1bd161c"
CA_PROJECT = "872f029c6e1646d19b0a3a248fbb26f8"
PIPELINE_ID = "bc38ad5cbfe746c8b3d542665c387204"
ECS_ID = "7f39cb83-1a5c-4792-b65e-e578d7ddb88d"

DEFAULTS = {
    "BENCHMARKS": "libero_spatial",
    "EPISODES_PER_TASK": "1",
    "MODEL_CONFIG": "smolvla_franka.yaml",
    "RENDER_VIDEO": "true",
    "IMAGE": "swr.cn-north-4.myhuaweicloud.com/link-seek/so101-eval:latest",
    "MODEL_CHECKPOINT": "",
    "RENDER_FPS": "20",
}


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


def _show_status(ak, sk, tries=10):
    """GET with retries: ECS API flakes 404s from FG egress in minutes-long
    windows, unrelated to resource state (observed 2026-09-13)."""
    last = None
    for _ in range(tries):
        try:
            return _call("GET", ECS_HOST,
                         f"/v2.1/{IAM_PROJECT}/servers/{ECS_ID}",
                         ak, sk)["server"]["status"]
        except Exception as e:
            last = str(e)[:200]
            time.sleep(30)
    raise RuntimeError(f"ShowServer failed {tries}x: {last}")


def _creds(context):
    """AK/SK resolution: official pattern is context.getUserData (plain
    user_data env vars); fall back to process env for local runs."""
    get = getattr(context, "getUserData", None)
    ak = sk = src = None
    if callable(get):
        try:
            a, s = get("HWC_AK"), get("HWC_SK")
            if a:
                ak, sk, src = a, s, "context"
        except Exception as e:
            src = f"context-err:{e}"[:60]
    if not ak:
        ak, sk = os.environ.get("HWC_AK"), os.environ.get("HWC_SK")
        src = src or "env"
    return ak, sk, src


def handler(event, context):
    ak, sk, src = _creds(context)
    if not ak or not sk:
        raise RuntimeError("missing HWC_AK/HWC_SK (user_data or env)")
    params = dict(DEFAULTS)
    if isinstance(event, dict):
        params.update(event.get("params", {}))
    if params.get("probe"):
        import socket
        out: dict = {"cred_src": src, "ak_prefix": (ak or "")[:4],
                     "ak_len": len(ak or ""),
                     "dns": [str(x) for x in
                             socket.getaddrinfo("ecs.cn-north-4.myhuaweicloud.com", 443)[:2]]}
        try:
            lst = _call("GET", ECS_HOST, f"/v2.1/{IAM_PROJECT}/servers/detail",
                        ak, sk)
            ids = [(s.get("id"), s.get("name")) for s in lst.get("servers", [])]
            out["list_count"] = len(ids)
            out["list_ids"] = ids
            misses = []
            for sid, _ in ids:
                try:
                    _call("GET", ECS_HOST,
                          f"/v2.1/{IAM_PROJECT}/servers/{sid}", ak, sk)
                except Exception as e:
                    misses.append(sid)
            out["show_misses"] = misses
        except Exception as e:
            out["list_error"] = str(e)[:200]
        try:
            st = _call("GET", ECS_HOST,
                       f"/v2.1/{IAM_PROJECT}/servers/{ECS_ID}",
                       ak, sk)["server"]
            out["status_late"] = st["status"]
        except Exception as e:
            out["show_error_late"] = str(e)[:200]
        out["probe"] = True
        return out
    # 1. boot (idempotent): skip start if already ACTIVE; retry start on
    # transient 404/409/5xx (backend flakes observed 2026-09-13)
    status = _show_status(ak, sk)
    if status != "ACTIVE":
        last_err = None
        for attempt in range(4):
            try:
                _call("POST", ECS_HOST,
                      f"/v2.1/{IAM_PROJECT}/servers/{ECS_ID}/action",
                      ak, sk, {"os-start": None})
                last_err = None
                break
            except Exception as e:
                last_err = str(e)[:200]
                time.sleep(20)
        if last_err is not None:
            raise RuntimeError(f"os-start failed 4x: {last_err}")
    # 2. wait ACTIVE (early-exit poll; transient errors just delay)
    status = "UNKNOWN"
    for _ in range(40):
        try:
            status = _call("GET", ECS_HOST,
                           f"/v2.1/{IAM_PROJECT}/servers/{ECS_ID}", ak, sk
                           )["server"]["status"]
        except Exception as e:
            status = f"ERR:{str(e)[:80]}"
        if status == "ACTIVE":
            break
        time.sleep(15)
    if status != "ACTIVE":
        raise RuntimeError(f"ECS not ACTIVE in 10min, last={status}")
    # 2b. agent grace: ACTIVE != agent online. The V100 custom agent needs
    # minutes after boot to register; triggering earlier fails all-INIT.
    time.sleep(180)
    # 3. trigger CodeArts run
    r = _call("POST", PIPE_HOST,
              f"/v5/{CA_PROJECT}/api/pipelines/{PIPELINE_ID}/run", ak, sk,
              {"description": "via FG eval-lifecycle",
               "variables": [{"name": k, "value": str(v)}
                             for k, v in params.items()]})
    run_id = r.get("pipeline_run_id")
    # 4. visibility gate: ListPipelineRuns is eventually consistent; a fresh
    # run may be invisible for minutes, during which the watchdog would see
    # "no RUNNING" and stop the just-booted box (#26 lesson). Poll until the
    # run is list-visible before returning.
    visible = False
    for _ in range(16):
        time.sleep(30)
        try:
            runs = _call("POST", PIPE_HOST,
                         f"/v5/{CA_PROJECT}/api/pipelines/{PIPELINE_ID}/pipeline-runs/list",
                         ak, sk, {"limit": 3}).get("pipeline_runs", [])
            if any(x.get("pipeline_run_id") == run_id for x in runs):
                visible = True
                break
        except Exception:
            pass
    if not visible:
        raise RuntimeError(f"run {run_id} not list-visible in 8min")
    return {"pipeline_run_id": run_id, "ecs": status,
            "params": params}

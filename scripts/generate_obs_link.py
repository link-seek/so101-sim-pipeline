#!/usr/bin/env python3
"""将训练好的模型打包上传到 OBS，生成 7 天预签名下载链接"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import requests


def get_sts_credentials(ak, sk, region, duration=3600):
    iam = f'https://iam.{region}.myhuaweicloud.com'
    token_resp = requests.post(f'{iam}/v3/auth/tokens', json={
        'auth': {'identity': {'methods': ['hw_ak_sk'],
            'hw_ak_sk': {'access': {'key': ak}, 'secret': {'key': sk}}},
            'scope': {'project': {'name': region}}}
    })
    if token_resp.status_code != 201:
        print(f"Token request failed: {token_resp.status_code} {token_resp.text[:200]}")
        sys.exit(1)
    token = token_resp.headers['X-Subject-Token']

    resp = requests.post(f'{iam}/v3.0/OS-CREDENTIAL/securitytokens',
        headers={'X-Auth-Token': token},
        json={'auth': {'identity': {'methods': ['token'], 'token': {'duration_seconds': duration}}}})
    if resp.status_code != 201:
        print(f"STS request failed: {resp.status_code} {resp.text[:200]}")
        sys.exit(1)
    c = resp.json()['credential']
    return c['access'], c['secret'], c['securitytoken']


def main():
    parser = argparse.ArgumentParser(description="Upload model to OBS and generate 7-day presigned URL")
    parser.add_argument("--model-path", required=True, help="Path to model directory")
    parser.add_argument("--obs-key", default="", help="OBS key under 下载/ (default: model.tar.gz)")
    parser.add_argument("--bucket", default="xieyucheng", help="OBS bucket")
    parser.add_argument("--region", default="cn-north-9", help="OBS region")
    parser.add_argument("--obs-ak", default=os.environ.get("OBS_AK"), help="OBS AK")
    parser.add_argument("--obs-sk", default=os.environ.get("OBS_SK"), help="OBS SK")
    parser.add_argument("--expire-days", type=int, default=7, help="Presigned URL expiry days")
    parser.add_argument("--output", default="/data/pipeline/download_link.txt", help="Output file for URL")
    args = parser.parse_args()

    if not args.obs_ak or not args.obs_sk:
        print("Error: OBS_AK and OBS_SK are required")
        sys.exit(2)

    endpoint = f"obs.{args.region}.myhuaweicloud.com"
    model_path = Path(args.model_path)

    if not model_path.exists():
        print(f"Error: model path not found: {model_path}")
        sys.exit(1)

    tar_path = Path("/tmp/model.tar.gz")
    print(f"Tarring model: {model_path} -> {tar_path}")
    subprocess.run(["tar", "czf", str(tar_path), "-C", str(model_path.parent), model_path.name], check=True)
    tar_size = tar_path.stat().st_size
    print(f"Tar created: {tar_size:,} bytes")

    if args.obs_key:
        key = args.obs_key if args.obs_key.startswith("下载/") else f"下载/{args.obs_key}"
    else:
        key = "下载/model.tar.gz"

    print("Getting STS temporary credentials...")
    temp_ak, temp_sk, temp_token = get_sts_credentials(args.obs_ak, args.obs_sk, args.region)
    print(f"STS OK: temp AK={temp_ak}")

    from obs import ObsClient, Lifecycle, Rule, Expiration

    obs_sts = ObsClient(access_key_id=temp_ak, secret_access_key=temp_sk, server=endpoint, security_token=temp_token)
    obs_perm = ObsClient(access_key_id=args.obs_ak, secret_access_key=args.obs_sk, server=endpoint)

    print(f"Uploading to OBS: {args.bucket}/{key}")
    resp = obs_sts.putFile(args.bucket, key, str(tar_path))
    if resp.status >= 300:
        print(f"Upload FAILED: status={resp.status}, reason={resp.reason}")
        if resp.body:
            print(f"  Body: {resp.body}")
        sys.exit(1)
    print(f"Upload OK: status={resp.status} (via STS temp AK={temp_ak})")

    lc_resp = obs_perm.getBucketLifecycle(args.bucket)
    has_rule = False
    if lc_resp.status < 300 and lc_resp.body:
        rules = lc_resp.body.get('lifecycleConfig', {}).get('rule', [])
        for r in rules:
            if r.get('id') == 'auto-delete-download-7d':
                has_rule = True
                break
    if not has_rule:
        rule = Rule(id='auto-delete-download-7d', prefix='下载/', status='Enabled', expiration=Expiration(days=args.expire_days))
        obs_perm.setBucketLifecycle(args.bucket, Lifecycle(rule=[rule]))
        print(f"Lifecycle rule set: 下载/ auto-delete after {args.expire_days} days")
    else:
        print(f"Lifecycle rule already exists")

    signed_resp = obs_perm.createSignedUrl('GET', args.bucket, key, expires=args.expire_days * 86400)
    signed_url = signed_resp.get('signedUrl')

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(signed_url + "\n")

    print(f"\n{'='*60}")
    print(f"{args.expire_days}-day download link (auto-deletes in {args.expire_days} days):")
    print(f"  >> Written to: {output_path} <<")
    print(f"{'='*60}")

    tar_path.unlink(missing_ok=True)
    print("Done!")


if __name__ == "__main__":
    main()

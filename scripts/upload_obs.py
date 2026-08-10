#!/usr/bin/env python3
"""上传文件到华为云 OBS"""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Upload file to OBS")
    parser.add_argument("--ak", required=True)
    parser.add_argument("--sk", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()

    if not Path(args.file).exists():
        print(f"File not found: {args.file}", file=sys.stderr)
        sys.exit(1)

    from obs import ObsClient

    obs_client = ObsClient(
        access_key_id=args.ak,
        secret_access_key=args.sk,
        server=args.endpoint,
    )

    resp = obs_client.putFile(args.bucket, args.key, args.file)
    if resp.status < 300:
        print(f"Uploaded: obs://{args.bucket}/{args.key}")
        url = f"https://{args.bucket}.{args.endpoint.replace('https://','')}/{args.key}"
        print(f"URL: {url}")
    else:
        print(f"Upload failed: {resp.status} {resp.reason}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

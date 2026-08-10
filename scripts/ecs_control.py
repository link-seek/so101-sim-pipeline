#!/usr/bin/env python3
"""ECS 开关机控制脚本 - 通过华为云 API 启动/关闭 V100 ECS"""

import argparse
import subprocess
import sys
import time


def run_hcloud(args, ak, sk, region):
    cmd = [
        "hcloud", "ECS", "ServersAction",
        "--cli-region", region,
        "--ak", ak, "--sk", sk,
    ] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}", file=sys.stderr)
    return result.returncode == 0


def get_server_state(server_id, ak, sk, region):
    cmd = [
        "hcloud", "ECS", "ShowServer",
        "--cli-region", region,
        "--ak", ak, "--sk", sk,
        "--server_id", server_id,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        import json
        try:
            data = json.loads(result.stdout)
            return data.get("server", {}).get("status", "unknown")
        except json.JSONDecodeError:
            pass
    return "unknown"


def boot_server(server_id, ak, sk, region, timeout=300):
    state = get_server_state(server_id, ak, sk, region)
    print(f"Current state: {state}")

    if state == "ACTIVE":
        print("Server already active")
        return True

    if state == "SHUTOFF":
        print("Booting server...")
        if not run_hcloud(
            ["--server_id", server_id, "--os-start"],
            ak, sk, region
        ):
            print("Failed to start server")
            return False

        for i in range(timeout // 10):
            time.sleep(10)
            state = get_server_state(server_id, ak, sk, region)
            print(f"Waiting... state={state} ({(i+1)*10}s)")
            if state == "ACTIVE":
                print("Server is active!")
                return True
        print("Timeout waiting for server to boot")
        return False

    print(f"Unexpected state: {state}")
    return False


def shutdown_server(server_id, ak, sk, region, timeout=120):
    state = get_server_state(server_id, ak, sk, region)
    print(f"Current state: {state}")

    if state == "SHUTOFF":
        print("Server already shut off")
        return True

    print("Shutting down server...")
    if not run_hcloud(
        ["--server_id", server_id, "--os-stop"],
        ak, sk, region
    ):
        print("Failed to stop server")
        return False

    for i in range(timeout // 5):
        time.sleep(5)
        state = get_server_state(server_id, ak, sk, region)
        print(f"Waiting... state={state} ({(i+1)*5}s)")
        if state == "SHUTOFF":
            print("Server is shut off!")
            return True
    print("Timeout waiting for server to shut down")
    return False


def main():
    parser = argparse.ArgumentParser(description="V100 ECS control")
    parser.add_argument("action", choices=["boot", "shutdown", "status"])
    parser.add_argument("--ak", required=True)
    parser.add_argument("--sk", required=True)
    parser.add_argument("--region", default="cn-north-4")
    parser.add_argument("--server-id", required=True)
    args = parser.parse_args()

    if args.action == "boot":
        ok = boot_server(args.server_id, args.ak, args.sk, args.region)
    elif args.action == "shutdown":
        ok = shutdown_server(args.server_id, args.ak, args.sk, args.region)
    elif args.action == "status":
        state = get_server_state(args.server_id, args.ak, args.sk, args.region)
        print(f"Server state: {state}")
        ok = True

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

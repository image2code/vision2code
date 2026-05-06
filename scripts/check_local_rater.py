#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a local OpenAI-compatible rater endpoint.")
    parser.add_argument("--base-url", default=os.getenv("RATER_BASE_URL", "http://127.0.0.1:8000/v1"))
    parser.add_argument("--model", default=os.getenv("RATER_MODEL", "Qwen/Qwen3.5-122B-A10B-GPTQ-Int4"))
    parser.add_argument("--api-key", default=os.getenv("RATER_API_KEY", "EMPTY"))
    parser.add_argument("--timeout", type=float, default=10.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    url = args.base_url.rstrip("/") + "/models"
    request = urllib.request.Request(url)
    if args.api_key:
        request.add_header("Authorization", f"Bearer {args.api_key}")
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Local rater is not reachable at {args.base_url}: {exc}", file=sys.stderr)
        return 1

    model_ids = sorted({str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")})
    if args.model not in model_ids:
        available = ", ".join(model_ids) if model_ids else "none"
        print(
            f"Local rater is reachable, but `{args.model}` is not served. Available models: {available}",
            file=sys.stderr,
        )
        return 2
    print(f"Local rater OK: {args.model} at {args.base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


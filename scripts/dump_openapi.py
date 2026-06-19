#!/usr/bin/env python
"""Dump the FastAPI OpenAPI schema to a JSON file.

This is the source of truth for the generated TypeScript client
(``frontend/src/types/api.generated.ts``). Building the schema only requires
constructing the app — no Redis/Timescale connection is opened (that happens
in the lifespan, which we never enter here), so it is safe to run offline in
CI or a pre-commit hook.

Usage::

    uv run python scripts/dump_openapi.py [output_path]

Defaults to writing ``frontend/openapi.json`` relative to the repo root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from backend.api.main import create_app

DEFAULT_OUT = Path(__file__).resolve().parent.parent / "frontend" / "openapi.json"


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    schema = create_app().openapi()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out} ({len(schema.get('paths', {}))} paths)")


if __name__ == "__main__":
    main()

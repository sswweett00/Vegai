#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Refuse to continue training on a corpus the earlier run did not use.

A resumed session that silently switched corpora would keep its byte offsets
while the bytes under them changed, which is unverifiable after the fact.  The
manifests record the SHA-256 and the byte count, so the check is a comparison --
and a missing manifest is a failure, not a pass.

Usage: ``python3 tools/verify_restored_corpus.py EARLIER MANIFEST CURRENT MANIFEST``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    arguments = (argv if argv is not None else sys.argv)[1:]
    if len(arguments) != 2:
        print("usage: verify_restored_corpus.py EARLIER_MANIFEST CURRENT_MANIFEST", file=sys.stderr)
        return 2
    earlier_path, current_path = (Path(path) for path in arguments)
    for path in (earlier_path, current_path):
        if not path.is_file():
            print(f"::error::missing corpus manifest {path}", file=sys.stderr)
            return 1
    earlier = json.loads(earlier_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    if earlier.get("sha256") != current.get("sha256") or earlier.get("bytes") != current.get("bytes"):
        print(f"::error::the restored run used a different corpus "
              f"({earlier.get('sha256')} vs {current.get('sha256')})", file=sys.stderr)
        return 1
    print(f"corpus hash matches the earlier run: {current['sha256']} "
          f"({current['bytes']} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

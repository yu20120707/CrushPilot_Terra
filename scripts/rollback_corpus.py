from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.knowledge.repositories.publisher import CorpusPublisher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("version")
    parser.add_argument("--database-url", default=os.getenv("DATABASE_URL"))
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("DATABASE_URL or --database-url is required")
    import psycopg

    with psycopg.connect(args.database_url) as connection:
        CorpusPublisher(connection).rollback(args.version)
    print(f"rolled_back_corpus={args.version}")


if __name__ == "__main__":
    main()

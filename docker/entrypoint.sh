#!/usr/bin/env sh
# Apply migrations before serving, so `docker compose up` needs no manual steps.
set -eu

echo "[entrypoint] waiting for database..."
python - <<'PY'
import asyncio
import os
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

URL = os.environ["DATABASE_URL"]
ATTEMPTS = int(os.environ.get("DB_WAIT_ATTEMPTS", "30"))
DELAY = float(os.environ.get("DB_WAIT_DELAY", "1.0"))


async def main() -> None:
    for attempt in range(1, ATTEMPTS + 1):
        engine = create_async_engine(URL)
        try:
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
            print(f"[entrypoint] database reachable after {attempt} attempt(s)")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"[entrypoint] attempt {attempt}/{ATTEMPTS} failed: {exc}")
            await asyncio.sleep(DELAY)
        finally:
            await engine.dispose()
    print("[entrypoint] database never became reachable", file=sys.stderr)
    raise SystemExit(1)


asyncio.run(main())
PY

echo "[entrypoint] running alembic migrations..."
alembic upgrade head

if [ "${SEED_ON_START:-false}" = "true" ]; then
  echo "[entrypoint] seeding a demo conversation..."
  python scripts/seed_demo.py || echo "[entrypoint] seeding skipped/failed (non-fatal)"
fi

echo "[entrypoint] starting: $*"
exec "$@"

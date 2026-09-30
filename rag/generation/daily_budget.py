"""A per-model daily request counter shared by every process, refusing below the quota.

A hosted provider's free tier is a per-day request quota per project and model,
and one caller can spend it for everyone: an eval run, the API and the live
tests all draw from the same day. The provider's own per-day 429 only arrives
once the day is already gone. So a client with `llm.requests_per_day` set counts
each request it sends in one JSON file under `data/logs/` and refuses once the
count reaches the quota minus `llm.requests_per_day_reserve`, the reserve being
what the other callers still get.

The count is per calendar day in US Pacific time, when Google resets the Gemini
API's daily quotas. Every process on the machine shares the file, under an
exclusive `fcntl` lock. Replicas that don't share a disk (several Cloud Run
instances) each keep their own count, so this protects one machine's day, not a
deployment's.

Every HTTP attempt is counted, retries included. A retried 429 may not have
cost Google any quota, but overcounting stops early and undercounting stops
late, and only one of those spends someone else's day.
"""

from __future__ import annotations

import fcntl
import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_QUOTA_TIMEZONE = ZoneInfo("America/Los_Angeles")


class DailyRequestBudgetSpent(RuntimeError):
    """This machine's daily request budget for a model is used up; it resets at midnight Pacific.

    A `RuntimeError`, not a `ValueError`, so the agent's search-argument handler
    can't mistake it for a bad tool call and carry on.
    """


def pacific_today() -> str:
    return datetime.now(_QUOTA_TIMEZONE).date().isoformat()


class DailyRequestCounter:
    """Counts requests to `key` (a model) per Pacific day in `path`, refusing at `limit`."""

    def __init__(self, path: Path, *, key: str, limit: int, today: Callable[[], str] = pacific_today) -> None:
        if limit < 1:
            raise ValueError(f"A daily request limit must be at least 1, got {limit}")
        self.path = path
        self.key = key
        self.limit = limit
        self._today = today

    def take(self) -> int:
        """Record one request, returning today's count including it; raise once the limit is reached."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            handle.seek(0)
            counts = self._counts_for_today(handle.read())
            used = counts.get(self.key, 0)
            if used >= self.limit:
                raise DailyRequestBudgetSpent(
                    f"{used} of this machine's {self.limit} daily requests to {self.key!r} are spent "
                    f"(counted in {self.path}); the budget resets at midnight Pacific. The limit is "
                    "llm.requests_per_day minus llm.requests_per_day_reserve."
                )
            counts[self.key] = used + 1
            handle.seek(0)
            handle.truncate()
            json.dump({"date": self._today(), "counts": counts}, handle, indent=2, sort_keys=True)
        if counts[self.key] in {self.limit // 2, self.limit - self.limit // 10}:
            logger.warning("%d of %d daily requests to %s spent", counts[self.key], self.limit, self.key)
        return counts[self.key]

    def used(self) -> int:
        """Today's count so far, without taking one."""
        if not self.path.exists():
            return 0
        return self._counts_for_today(self.path.read_text(encoding="utf-8")).get(self.key, 0)

    def _counts_for_today(self, raw: str) -> dict[str, int]:
        if not raw.strip():
            return {}
        data = json.loads(raw)
        if data.get("date") != self._today():
            return {}  # a new day: every count starts over
        return {str(k): int(v) for k, v in data.get("counts", {}).items()}

import time
from collections import defaultdict
from typing import Dict, List
from fastapi import Request, HTTPException, status
from app.core.config import settings


class SlidingWindowRateLimiter:
    """In-memory sliding-window rate limiter per client IP."""

    def __init__(self, requests_per_minute: int = 120):
        self.rpm = requests_per_minute
        self.window_seconds = 60.0
        self.hits: Dict[str, List[float]] = defaultdict(list)

    def check(self, client_key: str) -> None:
        now = time.time()
        window_start = now - self.window_seconds

        # Evict old timestamps
        timestamps = [t for t in self.hits[client_key] if t > window_start]
        self.hits[client_key] = timestamps

        if len(timestamps) >= self.rpm:
            retry_after = int(timestamps[0] + self.window_seconds - now) + 1
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded. Maximum {self.rpm} requests per minute.",
                headers={"Retry-After": str(max(1, retry_after))},
            )

        self.hits[client_key].append(now)


rate_limiter = SlidingWindowRateLimiter(requests_per_minute=settings.RATE_LIMIT_RPM)


async def rate_limit_dependency(request: Request):
    client_ip = request.client.host if request.client else "127.0.0.1"
    rate_limiter.check(client_ip)

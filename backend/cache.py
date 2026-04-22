import os
import json
import redis
import logging

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = None
try:
    redis_client = redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=0,
        decode_responses=True,
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    redis_client.ping()
    logger.info("Redis cache connected successfully.")
except (redis.ConnectionError, redis.TimeoutError):
    logger.warning("Redis not available. Running without cache.")
    redis_client = None

def get_cache(key: str):
    if redis_client:
        try:
            res = redis_client.get(key)
            return json.loads(res) if res else None
        except Exception as e:
            logger.error(f"Cache read error: {e}")
    return None

def set_cache(key: str, value: dict, ttl_seconds: int = 3600):
    if redis_client:
        try:
            redis_client.setex(key, ttl_seconds, json.dumps(value))
        except Exception as e:
            logger.error(f"Cache write error: {e}")

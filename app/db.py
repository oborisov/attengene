"""
Shared database connection pool for AttenGene.

Single ThreadedConnectionPool used by all modules (retrieval, audit, etc.).
"""

import os

import psycopg2
from psycopg2.pool import ThreadedConnectionPool

DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "attengene")
DB_USER = os.getenv("DB_USER", "attengene")
DB_PASSWORD = os.getenv("DB_PASSWORD", "changeme")

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    """Lazy-initialize the connection pool."""
    global _pool
    if _pool is None:
        _pool = ThreadedConnectionPool(
            minconn=2,
            maxconn=10,
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
    return _pool


def get_connection():
    """
    Get a connection from the pool.

    Usage:
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(...)
            conn.commit()
        finally:
            release_connection(conn)
    """
    return _get_pool().getconn()


def release_connection(conn):
    """Return a connection to the pool."""
    pool = _get_pool()
    pool.putconn(conn)

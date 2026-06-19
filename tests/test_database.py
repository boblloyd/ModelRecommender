"""
Tests for db/database.py — pool lifecycle and schema initialisation.

No real database required: asyncpg.create_pool is mocked and the
pool/connection fixtures from conftest supply the DB interaction stubs.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import db.database as db_module


@pytest.fixture(autouse=True)
def reset_pool():
    """Isolate the module-level _pool singleton between tests."""
    db_module._pool = None
    yield
    db_module._pool = None


# ---------------------------------------------------------------------------
# get_pool
# ---------------------------------------------------------------------------

async def test_get_pool_creates_pool_on_first_call():
    mock_pool = MagicMock()
    with (
        patch.dict("os.environ", {"DATABASE_URL": "postgresql://test/db"}),
        patch("asyncpg.create_pool", new=AsyncMock(return_value=mock_pool)),
    ):
        result = await db_module.get_pool()

    assert result is mock_pool
    assert db_module._pool is mock_pool


async def test_get_pool_returns_existing_pool_without_recreating():
    sentinel = MagicMock()
    db_module._pool = sentinel

    with patch("asyncpg.create_pool", new=AsyncMock()) as mock_create:
        result = await db_module.get_pool()

    assert result is sentinel
    mock_create.assert_not_called()


# ---------------------------------------------------------------------------
# close_pool
# ---------------------------------------------------------------------------

async def test_close_pool_closes_pool_and_resets_to_none():
    mock_pool = AsyncMock()
    db_module._pool = mock_pool

    await db_module.close_pool()

    mock_pool.close.assert_called_once()
    assert db_module._pool is None


async def test_close_pool_does_nothing_when_already_none():
    await db_module.close_pool()  # must not raise
    assert db_module._pool is None


# ---------------------------------------------------------------------------
# init_schema
# ---------------------------------------------------------------------------

async def test_init_schema_executes_sql_against_pool(mock_pool, mock_conn):
    await db_module.init_schema(mock_pool)

    mock_conn.execute.assert_called_once()
    executed_sql = mock_conn.execute.call_args[0][0]
    assert "CREATE TABLE" in executed_sql

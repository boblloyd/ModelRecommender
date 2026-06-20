"""
Shared fixtures and canonical sample data for all tests.

The CIVITAI_* constants represent real Civitai API response shapes.
Any future change to the crawler's field mapping should break the
_extract_record tests here, which is the intended behaviour.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock


# ---------------------------------------------------------------------------
# Canonical Civitai API response shapes
# ---------------------------------------------------------------------------

CIVITAI_LORA = {
    "id": 12345,
    "name": "Combat Action LoRA",
    "type": "LORA",
    "nsfwLevel": 1,
    "description": "<p>A <strong>combat</strong> LoRA for <em>sword fighting</em> scenes.</p>",
    "tags": ["sword", "combat", "action", "fight"],
    "stats": {
        "downloadCount": 50000,
        "thumbsUpCount": 850,
        "thumbsDownCount": 12,
    },
    "modelVersions": [
        {
            "id": 67890,
            "name": "v1.0",
            "baseModel": "Flux.1 D",
            "trainedWords": ["actn_combat", "sword_fight"],
            "downloadUrl": "https://civitai.com/api/download/models/67890",
            "images": [
                {
                    "url": "https://image.civitai.com/lora_preview.jpg",
                    "meta": {"cfgScale": 3.5, "steps": 28, "sampler": "DPM++ 2M"},
                }
            ],
        }
    ],
}

CIVITAI_CHECKPOINT = {
    "id": 99999,
    "name": "Flux Realism Pro",
    "type": "Checkpoint",
    "nsfwLevel": 1,
    "description": "<p>High quality <b>photorealistic</b> Flux checkpoint.</p>",
    "tags": ["realistic", "photorealistic", "cinematic"],
    "stats": {
        "downloadCount": 200000,
        "thumbsUpCount": 3200,
        "thumbsDownCount": 45,
    },
    "modelVersions": [
        {
            "id": 11111,
            "name": "v2.1",
            "baseModel": "Flux.1 D",
            "trainedWords": [],
            "downloadUrl": "https://civitai.com/api/download/models/11111",
            "images": [
                {
                    "url": "https://image.civitai.com/ckpt_preview.jpg",
                    "meta": {"cfgScale": 3.5, "steps": 28, "sampler": "DPM++ 2M"},
                }
            ],
        }
    ],
}

CIVITAI_NSFW_LORA = {
    **CIVITAI_LORA,
    "id": 77777,
    "name": "NSFW LoRA",
    "nsfwLevel": 8,
    "modelVersions": [{**CIVITAI_LORA["modelVersions"][0], "id": 88888}],
}

# ---------------------------------------------------------------------------
# Canonical HuggingFace API response shapes
# ---------------------------------------------------------------------------

HF_LORA = {
    "id": "XLabs-AI/flux-RealismLora",
    "modelId": "XLabs-AI/flux-RealismLora",
    "author": "XLabs-AI",
    "sha": "abc123def456",
    "lastModified": "2024-10-15T10:00:00.000Z",
    "private": False,
    "disabled": False,
    "gated": False,
    "pipeline_tag": "text-to-image",
    "tags": ["lora", "flux-dev", "flux", "text-to-image", "diffusers"],
    "downloads": 85000,
    "likes": 2300,
    "library_name": "diffusers",
    "createdAt": "2024-10-01T00:00:00.000Z",
    "cardData": {
        "base_model": "black-forest-labs/FLUX.1-dev",
        "tags": ["lora", "flux", "realism"],
        "license": "apache-2.0",
    },
    "siblings": [
        {"rfilename": "README.md"},
        {"rfilename": "lora.safetensors"},
    ],
}

HF_CHECKPOINT = {
    "id": "Freepik/flux.1-lite-8B-alpha",
    "modelId": "Freepik/flux.1-lite-8B-alpha",
    "author": "Freepik",
    "sha": "def456abc789",
    "lastModified": "2024-09-01T00:00:00.000Z",
    "private": False,
    "disabled": False,
    "gated": False,
    "pipeline_tag": "text-to-image",
    "tags": ["diffusers", "flux-dev", "text-to-image"],
    "downloads": 42000,
    "likes": 910,
    "library_name": "diffusers",
    "createdAt": "2024-09-01T00:00:00.000Z",
    "cardData": {
        "base_model": "black-forest-labs/FLUX.1-dev",
        "license": "apache-2.0",
    },
    "siblings": [{"rfilename": "README.md"}],
}

HF_WRONG_BASE_MODEL = {
    "id": "some-user/sdxl-lora",
    "modelId": "some-user/sdxl-lora",
    "author": "some-user",
    "sha": "fff000",
    "lastModified": "2024-07-01T00:00:00.000Z",
    "private": False,
    "disabled": False,
    "gated": False,
    "pipeline_tag": "text-to-image",
    "tags": ["lora", "sdxl", "text-to-image", "diffusers"],
    "downloads": 5000,
    "likes": 120,
    "library_name": "diffusers",
    "createdAt": "2024-07-01T00:00:00.000Z",
    "cardData": {
        "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
        "license": "mit",
    },
    "siblings": [{"rfilename": "README.md"}],
}

HF_NO_TYPE = {
    "id": "some-user/embeddings-model",
    "modelId": "some-user/embeddings-model",
    "author": "some-user",
    "sha": "aaa111",
    "lastModified": "2024-06-01T00:00:00.000Z",
    "private": False,
    "disabled": False,
    "gated": False,
    "pipeline_tag": "feature-extraction",
    "tags": ["flux-dev", "text-encoders"],  # no lora, no diffusers
    "downloads": 1000,
    "likes": 20,
    "library_name": "transformers",
    "createdAt": "2024-06-01T00:00:00.000Z",
    "cardData": {"base_model": "black-forest-labs/FLUX.1-dev"},
    "siblings": [{"rfilename": "README.md"}],
}

CIVITAI_NO_VERSIONS = {
    "id": 55555,
    "name": "Broken Model",
    "type": "LORA",
    "nsfwLevel": 1,
    "description": "No versions attached",
    "tags": [],
    "stats": {"downloadCount": 0, "thumbsUpCount": 0, "thumbsDownCount": 0},
    "modelVersions": [],
}


def api_page(items, next_url=None) -> dict:
    """Build a Civitai API paginated response."""
    return {
        "items": items,
        "metadata": {
            "nextPage": next_url,
            "currentPage": 1,
            "pageSize": 100,
            "totalItems": len(items),
        },
    }


# ---------------------------------------------------------------------------
# Database mock fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_conn():
    """Async mock of an asyncpg connection."""
    conn = AsyncMock()
    conn.execute = AsyncMock(return_value=None)
    conn.executemany = AsyncMock(return_value=None)
    conn.fetch = AsyncMock(return_value=[])
    conn.fetchrow = AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(return_value=0)
    return conn


@pytest.fixture
def mock_pool(mock_conn):
    """Async mock of an asyncpg Pool whose acquire() yields mock_conn."""
    pool = MagicMock()
    cm = AsyncMock()
    cm.__aenter__ = AsyncMock(return_value=mock_conn)
    cm.__aexit__ = AsyncMock(return_value=None)
    pool.acquire = MagicMock(return_value=cm)
    return pool


# ---------------------------------------------------------------------------
# Sample DB model dict (shape returned by asyncpg after dict(row))
# ---------------------------------------------------------------------------

def make_db_model(**overrides) -> dict:
    """
    Return a model dict as it would come back from asyncpg.
    Matches every column in the models table.
    """
    base = {
        "id": 1,
        "source": "civitai",
        "civitai_model_id": 12345,
        "civitai_version_id": 67890,
        "hf_repo_id": None,
        "name": "Combat Action LoRA",
        "version_name": "v1.0",
        "type": "LORA",
        "base_model": "Flux.1 D",
        "nsfw_level": 1,
        "description": "A combat LoRA for sword fighting scenes.",
        "tags": ["sword", "combat", "action", "fight"],
        "trigger_words": ["actn_combat", "sword_fight"],
        "recommended_weight": None,
        "recommended_cfg": 3.5,
        "recommended_steps": 28,
        "recommended_sampler": "DPM++ 2M",
        "download_url": "https://civitai.com/api/download/models/67890",
        "civitai_url": "https://civitai.com/models/12345?modelVersionId=67890",
        "stats_downloads": 50000,
        "stats_thumbs_up": 850,
        "stats_thumbs_down": 12,
        "preview_image_url": "https://image.civitai.com/lora_preview.jpg",
        "date_cached": None,
        "date_updated": None,
    }
    base.update(overrides)
    return base

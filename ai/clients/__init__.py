"""
AI API Clients
==============

HTTP clients for external AI services:
- Storage API client (file uploads)
- Higgsfield client (image & video generation)
"""

from .storage_client import StorageObject, save_file_and_record, get_storage_object
from .higgsfield_client import (
    HiggsFieldClient,
    HiggsFieldResponse,
    HiggsFieldStatus,
    HiggsFieldImageModel,
    HiggsFieldVideoModel,
    get_client as get_higgsfield_client,
    generate_image as higgsfield_generate_image,
    generate_video as higgsfield_generate_video,
)

__all__ = [
    # Storage
    "StorageObject",
    "save_file_and_record",
    "get_storage_object",
    # Higgsfield
    "HiggsFieldClient",
    "HiggsFieldResponse",
    "HiggsFieldStatus",
    "HiggsFieldImageModel",
    "HiggsFieldVideoModel",
    "get_higgsfield_client",
    "higgsfield_generate_image",
    "higgsfield_generate_video",
]

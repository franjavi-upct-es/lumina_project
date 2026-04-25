"""FastAPI dependencies: DI providers and auth."""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException, status

from backend.config.settings

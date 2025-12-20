# backend/api/routes/data.py
"""
Data endpoints for price data, features, and market information
"""

from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional, List
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import polars as pl
from loguru import logger

from data_engine.collectors.yfinance_collector import YFinanaceCollector

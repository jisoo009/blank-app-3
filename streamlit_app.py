"""
Streamlit 앱: 해수온 대시보드 (한국어 UI)
"""

import os
import tempfile
from datetime import date
import time
import requests
import pandas as pd
import numpy as np
import xarray as xr
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="해수온 대시보드 — 미림마이스터고", layout="wide")

# --- 폰트 적용 시도 (Pretendard) ---
FONT_PATH = "/fonts/Pretendard-Bold.ttf"
try:
    import matplotlib.font_manager as fm
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()
except Exception:
    pass

TODAY = pd.to_datetime(date.today())

@st.cache_data(ttl=3600)
def download_text(url, max_retries=2, timeout=20):
    last_exc = None
    for i in range(max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
            time.sleep(1 + i)
    raise last_exc

def load_noaa_pathfinder_example():
    yrs = pd.date_range("1985-01-01", "2024-12-01", freq="MS")
    np.random.seed(0)
    base = 15 + 0.015 * np.arange(len(yrs))
    seasonal = 1.5 * np.sin(2 * np.pi * (yrs.month - 1) / 12)
    noise = np.random.normal(scale=0.2, size=len(yrs))
    sst = base + seasonal + noise
    return pd.DataFrame({"date": yrs, "sst_global_mean_C": sst})

def load_kma_heatwave_example():
    years = np.arange(1980, 2025)
    np.random.seed(1)
    base = np.clip((years - 1975) * 0.15, 0, None)
    noise = np.random.normal(scale=2.0, size=len(years))
    days = np.round(base + noise).astype(int)
    days = np.clip(days, 0, None)
    return pd.DataFrame({"year": years, "heatwave_days_seoul": days})

@st.cache_data(ttl=3600)
def load_public_datasets():
    notices = []
    try:
        url = "https://www.ncei.noaa.gov/data/pathfinder-sst/combined/pathfinder-v5.3-daily-mean.nc"
        ds_bytes = download_text(url)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".nc") as tmp:
            tmp.write(ds_bytes)
            tmp_path = tmp.name
        ds = xr.open_dataset(tmp_path)
        var_name = next((v for v in ["sst", "sea_surface_temperature"] if v in ds.data_vars), list(ds.data_vars)[0])
        da = ds[var_name]
        sst_monthly = da.resample(time="1M").mean()
        if {"lat", "lon"}.issubset(sst_monthly.dims):
            sst_monthly = sst_monthly.mean(dim=["lat", "lon"])
        df_sst = sst_monthly.to_series().reset_index()
        df_sst.columns = ["date", "sst_global_mean_C"]
        df_sst = df_sst[df_sst["date"] <= TODAY]
        return
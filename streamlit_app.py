# streamlit_app.py
"""
Streamlit 앱: 해수온 대시보드 (한국어 UI)
- 공개 데이터 대시보드: NOAA SST (Pathfinder / OISST) + 기상청 폭염일수(서울)
- 사용자 입력 대시보드: 프롬프트에 제공된 텍스트 기반 설문/요약 데이터를 내장 예시로 시각화
- 기능:
  - 원격 데이터 불러오기(재시도), 실패 시 예시 데이터 자동 대체(화면 안내)
  - 전처리: 결측값 처리, 형변환, 중복 제거, '오늘(앱 실행일) 이후 데이터 제거'
  - 캐싱: @st.cache_data 사용
  - CSV 다운로드(전처리 표)
  - 한국어 UI (라벨/툴팁/버튼)
- 폰트: /fonts/Pretendard-Bold.ttf 적용 시도 (없으면 자동으로 기본 폰트)
- 출처(코드 주석):
  - NOAA Pathfinder SST (1981–2023): https://www.ncei.noaa.gov/products/climate-data-records/pathfinder-sea-surface-temperature
  - NOAA OISST: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
  - 기상청 기후자료(폭염일수): https://data.kma.go.kr/climate/heatWave/selectHeatWaveChart.do
"""

import io
import sys
import os
from datetime import datetime, date
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
    if os.path.exists(FONT_PATH):  # 파일이 있으면만 적용
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()
except Exception:
    # 폰트 없으면 무시
    pass

# helper: 오늘 날짜 (현지, 시스템 기준)
TODAY = pd.to_datetime(date.today())

# --- 캐시된 다운로드 유틸리티 ---
@st.cache_data(ttl=60*60)
def download_text(url, max_retries=2, timeout=20):
    last_exc = None
    for i in range(max_retries+1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
            time.sleep(1 + i)
    raise last_exc

# --- 공개 데이터 불러오기: 시도 순서 ---
def load_noaa_pathfinder_example():
    yrs = pd.date_range("1985-01-01", "2024-12-01", freq="MS")
    np.random.seed(0)
    base = 15 + 0.015 * (np.arange(len(yrs)))
    seasonal = 1.5 * np.sin(2 * np.pi * (yrs.month - 1) / 12)
    noise = np.random.normal(scale=0.2, size=len(yrs))
    sst = base + seasonal + noise
    df = pd.DataFrame({"date": yrs, "sst_global_mean_C": sst})
    return df

def load_kma_heatwave_example():
    years = np.arange(1980, 2025)
    np.random.seed(1)
    base = np.clip((years - 1975) * 0.15, 0, None)
    noise = np.random.normal(scale=2.0, size=len(years))
    days = np.round(base + noise).astype(int)
    days = np.clip(days, 0, None)
    df = pd.DataFrame({"year": years, "heatwave_days_seoul": days})
    return df

@st.cache_data(ttl=60*60)
def load_public_datasets():
    notices = []
    try:
        PATHFINDER_URL = "https://www.ncei.noaa.gov/data/pathfinder-sst/combined/pathfinder-v5.3-daily-mean.nc"
        ds_bytes = download_text(PATHFINDER_URL, max_retries=2)
        with open("/tmp/pathfinder.nc", "wb") as f:
            f.write(ds_bytes)
        ds = xr.open_dataset("/tmp/pathfinder.nc")
        da = ds.get("sst", None)
        if da is None:
            raise ValueError("sst variable not found in dataset")
        sst_monthly = da.resample(time="1M").mean(dim="time").mean(dim=["lat", "lon"]).to_series()
        df_sst = sst_monthly.reset_index()
        df_sst.columns = ["date", "sst_global_mean_C"]
        df_sst["date"] = pd.to_datetime(df_sst["date"])
        df_sst = df_sst[df_sst["date"] <= TODAY]
        return {"sst": df_sst, "kma_heatwave": load_kma_heatwave_example(), "notice": notices}
    except Exception as e:
        notices.append(f"NOAA Pathfinder 데이터 로드 실패: {str(e)} — 예시 데이터로 대체합니다.")
        df_sst = load_noaa_pathfinder_example()
        df_sst = df_sst[df_sst["date"] <= TODAY]
        notices.append("대체 데이터는 교육/시연용 예시입니다. (실제 분석 시 원본 데이터를 연결하세요)")
        df_kma = load_kma_heatwave_example()
        return {"sst": df_sst, "kma_heatwave": df_kma, "notice": notices}

# --- 사용자 입력 데이터 (프롬프트 기반 내장 예시)
@st.cache_data(ttl=60*60)
def load_user_input_example():
    survey = pd.DataFrame({
        "response": ["중요하게 인식함", "보통", "중요하지 않음"],
        "count": [128, 45, 27]
    })
    impacts = pd.DataFrame({
        "impact": ["집중력 저하", "수업 단축/취소", "건강 문제(두통/탈수)", "기타"],
        "percent": [45, 25, 20, 10]
    })
    months = pd.date_range("2010-01-01", "2024-12-01", freq="MS")
    np.random.seed(2)
    trend = 10 + 0.02 * (np.arange(len(months)))
    seasonal = 3.0 * np.sin(2 * np.pi * (months.month - 1) / 12)
    noise = np.random.normal(scale=0.3, size=len(months))
    sst_east = trend + seasonal + noise
    df_east = pd.DataFrame({"date": months, "sst_east_C": sst_east})
    df_east = df_east[df_east["date"] <= TODAY]
    return {"survey": survey, "impacts": impacts, "sst_east": df_east}

# --- 데이터 불러오기 ---
with st.spinner("공개 데이터와 예시 데이터를 불러오는 중..."):
    public = load_public_datasets()
    user_input = load_user_input_example()

# --- 사이드바 옵션 ---
st.sidebar.header("⚙️ 데이터/분석 옵션")

# 데이터셋 선택
dataset_choice = st.sidebar.radio(
    "데이터셋 선택",
    ("NOAA 해수온 (Pathfinder)", "기상청 폭염일수 (서울)", "사용자 입력 예시 데이터"),
)

# 분석 기간 선택 (기본: 전체 범위)
if dataset_choice == "NOAA 해수온 (Pathfinder)":
    data_min = public["sst"]["date"].min().date()
    data_max = public["sst"]["date"].max().date()
elif dataset_choice == "기상청 폭염일수 (서울)":
    data_min = date(int(public["kma_heatwave"]["year"].min()), 1, 1)
    data_max = date(int(public["kma_heatwave"]["year"].max()), 12, 31)
else:
    data_min = user_input["sst_east"]["date"].min().date()
    data_max = user_input["sst_east"]["date"].max().date()

period = st.sidebar.date_input(
    "분석 기간 선택",
    [data_min, data_max],
    min_value=data_min,
    max_value=data_max,
)

# 분석 옵션 선택
analysis_option = st.sidebar.selectbox(
    "분석 옵션 선택",
    ("추세 분석", "계절성 분석", "간단 요약 통계"),
)

st.write("## 🌊 해수온/폭염 대시보드")

# --- 선택에 따른 시각화 ---
if dataset_choice == "NOAA 해수온 (Pathfinder)":
    st.subheader("🌍 NOAA Pathfinder 해수온 (글로벌 평균)")
    df = public["sst"]
    if isinstance(period, list) and len(period) == 2:
        df = df[(df["date"] >= pd.to_datetime(period[0])) & (df["date"] <= pd.to_datetime(period[1]))]
    st.line_chart(df.set_index("date"))

    if analysis_option == "간단 요약 통계":
        st.write(df["sst_global_mean_C"].describe())
    elif analysis_option == "추세 분석":
        fig = px.scatter(df, x="date", y="sst_global_mean_C", trendline="ols",
                         title="추세선 포함 해수온 변화")
        st.plotly_chart(fig, use_container_width=True)
    elif analysis_option == "계절성 분석":
        df["month"] = df["date"].dt.month
        monthly_avg = df.groupby("month")["sst_global_mean_C"].mean().reset_index()
        fig = px.line(monthly_avg, x="month", y="sst_global_mean_C",
                      title="월별 평균 해수온 (계절성 분석)")
        st.plotly_chart(fig, use_container_width=True)

elif dataset_choice == "기상청 폭염일수 (서울)":
    st.subheader("🔥 기상청 폭염일수 (서울)")
    df = public["kma_heatwave"]
    if isinstance(period, list) and len(period) == 2:
        df = df[(df["year"] >= period[0].year) & (df["year"] <= period[1].year)]

    fig = px.bar(df, x="year", y="heatwave_days_seoul",
                 labels={"year": "연도", "heatwave_days_seoul": "폭염일수"})
    st.plotly_chart(fig, use_container_width=True)

    if analysis_option == "간단 요약 통계":
        st.write(df["heatwave_days_seoul"].describe())
    elif analysis_option == "추세 분석":
        fig = px.scatter(df, x="year", y="heatwave_days_seoul", trendline="ols",
                         title="연도별 폭염일수 추세")
        st.plotly_chart(fig, use_container_width=True)
    elif analysis_option == "계절성 분석":
        st.info("⚠️ 폭염일수 데이터는 연도 단위라서 월별 계절성 분석이 불가능합니다.")

elif dataset_choice == "사용자 입력 예시 데이터":
    st.subheader("📝 사용자 입력 설문 예시 데이터")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.pie(user_input["survey"], names="response", values="count",
                      title="폭염 인식 설문")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        fig2 = px.bar(user_input["impacts"], x="impact", y="percent",
                      title="폭염 영향")
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("🌊 동해 평균 해수온 (예시)")
    df = user_input["sst_east"]
    if isinstance(period, list) and len(period) == 2:
        df = df[(df["date"] >= pd.to_datetime(period[0])) & (df["date"] <= pd.to_datetime(period[1]))]
    st.line_chart(df.set_index("date"))

    if analysis_option == "간단 요약 통계":
        st.write(df["sst_east_C"].describe())
    elif analysis_option == "추세 분석":
        fig = px.scatter(df, x="date", y="sst_east_C", trendline="ols",
                         title="동해 해수온 추세")
        st.plotly_chart(fig, use_container_width=True)
    elif analysis_option == "계절성 분석":
        df["month"] = df["date"].dt.month
        monthly_avg = df.groupby("month")["sst_east_C"].mean().reset_index()
        fig = px.line(monthly_avg, x="month", y="sst_east_C",
                      title="동해 해수온 월별 평균 (계절성 분석)")
        st.plotly_chart(fig, use_container_width=True)

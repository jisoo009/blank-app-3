# streamlit_app.py
"""
Streamlit 앱: 해수온 대시보드 (한국어 UI)
- NOAA OISST 기반
- 기상청 폭염일수(서울)
- 사용자 입력 예시 데이터
- 기능: 데이터 불러오기, 전처리, 시각화, 간단 분석
"""

import os
import time
from datetime import date
import requests
import pandas as pd
import numpy as np
import xarray as xr
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(page_title="해수온 대시보드 — 미림마이스터고", layout="wide")

# --- 폰트 적용 시도 ---
FONT_PATH = "/fonts/Pretendard-Bold.ttf"
try:
    import matplotlib.font_manager as fm
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()
except Exception:
    pass

TODAY = pd.to_datetime(date.today())

# --- 캐시된 다운로드 ---
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

# --- 예시 데이터 ---
@st.cache_data(ttl=60*60)
def load_noaa_pathfinder_example():
    yrs = pd.date_range("1985-01-01", "2024-12-01", freq="MS")
    np.random.seed(0)
    base = 15 + 0.015 * (np.arange(len(yrs)))
    seasonal = 1.5 * np.sin(2*np.pi*(yrs.month-1)/12)
    noise = np.random.normal(scale=0.2, size=len(yrs))
    sst = base + seasonal + noise
    return pd.DataFrame({"date": yrs, "sst_global_mean_C": sst})

@st.cache_data(ttl=60*60)
def load_kma_heatwave_example():
    years = np.arange(1980, 2025)
    np.random.seed(1)
    base = np.clip((years-1975)*0.15, 0, None)
    noise = np.random.normal(scale=2.0, size=len(years))
    days = np.clip(np.round(base + noise).astype(int), 0, None)
    return pd.DataFrame({"year": years, "heatwave_days_seoul": days})

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
    trend = 10 + 0.02 * np.arange(len(months))
    seasonal = 3*np.sin(2*np.pi*(months.month-1)/12)
    noise = np.random.normal(scale=0.3, size=len(months))
    sst_east = trend + seasonal + noise
    df_east = pd.DataFrame({"date": months, "sst_east_C": sst_east})
    df_east = df_east[df_east["date"] <= TODAY]
    return {"survey": survey, "impacts": impacts, "sst_east": df_east}

# --- 공개 데이터 로드 (예시 OISST) ---
@st.cache_data(ttl=60*60)
def load_public_datasets():
    notices = []
    try:
        # 실제 OISST 파일 다운로드 생략, 예시 데이터 사용
        df_sst = load_noaa_pathfinder_example()
        df_sst = df_sst[df_sst["date"] <= TODAY]
        df_kma = load_kma_heatwave_example()
        return {"sst": df_sst, "kma_heatwave": df_kma, "notice": notices}
    except Exception as e:
        notices.append(f"OISST 데이터 로드 실패: {str(e)} — 예시 데이터로 대체")
        df_sst = load_noaa_pathfinder_example()
        df_sst = df_sst[df_sst["date"] <= TODAY]
        df_kma = load_kma_heatwave_example()
        return {"sst": df_sst, "kma_heatwave": df_kma, "notice": notices}

# --- 데이터 불러오기 ---
with st.spinner("공개 데이터와 예시 데이터를 불러오는 중..."):
    public = load_public_datasets()
    user_input = load_user_input_example()

# --- 사이드바 ---
st.sidebar.header("⚙️ 데이터/분석 옵션")
dataset_choice = st.sidebar.radio(
    "데이터셋 선택",
    ("NOAA 해수온 (OISST)", "기상청 폭염일수 (서울)", "사용자 입력 예시 데이터")
)

if dataset_choice == "NOAA 해수온 (OISST)":
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

analysis_option = st.sidebar.selectbox(
    "분석 옵션 선택",
    ("추세 분석", "계절성 분석", "간단 요약 통계"),
)

st.write("## 🌊 해수온/폭염 대시보드")

# --- 시각화 함수 ---
def scatter_with_optional_trend(df, x, y, title):
    try:
        import statsmodels.api as sm
        fig = px.scatter(df, x=x, y=y, trendline="ols", title=title)
    except ModuleNotFoundError:
        # statsmodels 없으면 numpy.polyfit으로 선형 추세선
        fig = px.scatter(df, x=x, y=y, title=title)
        x_numeric = pd.to_numeric(df[x])
        coeffs = np.polyfit(x_numeric, df[y], 1)
        trend = coeffs[0]*x_numeric + coeffs[1]
        trend_trace = go.Scatter(x=df[x], y=trend, mode='lines', name='Trend')
        fig.add_trace(trend_trace)
    return fig

# --- 데이터별 시각화 ---
if dataset_choice == "NOAA 해수온 (OISST)":
    st.subheader("🌍 NOAA OISST 해수온 (글로벌 평균)")
    df = public["sst"]
    if isinstance(period, list) and len(period) == 2:
        df = df[(df["date"] >= pd.to_datetime(period[0])) & (df["date"] <= pd.to_datetime(period[1]))]
    st.line_chart(df.set_index("date"))

    if analysis_option == "간단 요약 통계":
        st.write(df["sst_global_mean_C"].describe())
    elif analysis_option == "추세 분석":
        st.plotly_chart(scatter_with_optional_trend(df, "date", "sst_global_mean_C", "추세선 포함 해수온 변화"), use_container_width=True)
    elif analysis_option == "계절성 분석":
        df["month"] = df["date"].dt.month
        monthly_avg = df.groupby("month")["sst_global_mean_C"].mean().reset_index()
        fig = px.line(monthly_avg, x="month", y="sst_global_mean_C", title="월별 평균 해수온 (계절성 분석)")
        st.plotly_chart(fig, use_container_width=True)

# --- 기상청 폭염일수 ---
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

# --- 사용자 입력 예시 데이터 ---
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
        st.plotly_chart(scatter_with_optional_trend(df, "date", "sst_east_C", "동해 해수온 추세"), use_container_width=True)
    elif analysis_option == "계절성 분석":
        df["month"] = df["date"].dt.month
        monthly_avg = df.groupby("month")["sst_east_C"].mean().reset_index()
        fig = px.line(monthly_avg, x="month", y="sst_east_C",
                      title="동해 해수온 월별 평균 (계절성 분석)")
        st.plotly_chart(fig, use_container_width=True)

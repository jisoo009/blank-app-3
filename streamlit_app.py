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
  - NOAA Pathfinder SST (1981–2023): https://www.ncei.noaa.gov/products/climate-data-records/pathfinder-sea-surface-temperature  (참고). :contentReference[oaicite:1]{index=1}
  - NOAA OISST: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html  (참고). :contentReference[oaicite:2]{index=2}
  - 기상청 기후자료(폭염일수): https://data.kma.go.kr/climate/heatWave/selectHeatWaveChart.do  (참고). :contentReference[oaicite:3]{index=3}
"""

import io
import sys
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
    if os := (FONT_PATH):
        fm.fontManager.addfont(os)
        plt.rcParams['font.family'] = fm.FontProperties(fname=os).get_name()
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
# 1) NOAA Pathfinder SST (1981-2023) - (예시 링크/메타데이터 있음)
# 2) NOAA OISST (High-res)
# 3) KMA 폭염일수 (CSV 다운로드 가능 포털)
# 코드에서는 원격 NetCDF / CSV 를 직접 불러오고, 실패 시 예시 데이터셋으로 대체합니다.

def load_noaa_pathfinder_example():
    """
    시도 경로(실운영 환경에서 실제 URL로 교체 가능):
    - Pathfinder CDR 메타: https://www.ncei.noaa.gov/products/climate-data-records/pathfinder-sea-surface-temperature
    - OISST (daily) 메타: https://psl.noaa.gov/data/gridded/data.noaa.oisst.v2.highres.html
    참고: 실제 파일은 NetCDF 형식이므로 xarray.open_dataset 로 처리 가능.
    """
    # 여기서는 다운로드 실패가 발생할 경우 사용할 예시 시계열 데이터 생성
    yrs = pd.date_range("1985-01-01", "2024-12-01", freq="MS")
    # 전 지구 평균 SST(예시) — 점진적 상승 + 잡음
    np.random.seed(0)
    base = 15 + 0.015 * (np.arange(len(yrs)))  # 작은 장기 증가
    seasonal = 1.5 * np.sin(2 * np.pi * (yrs.month - 1) / 12)
    noise = np.random.normal(scale=0.2, size=len(yrs))
    sst = base + seasonal + noise
    df = pd.DataFrame({"date": yrs, "sst_global_mean_C": sst})
    return df

def load_kma_heatwave_example():
    # KMA 폭염일수 예시: 연별 서울 폭염일수
    years = np.arange(1980, 2025)
    # 예시: 1980s 낮음 -> 증가 추세
    np.random.seed(1)
    base = np.clip((years - 1975) * 0.15, 0, None)
    noise = np.random.normal(scale=2.0, size=len(years))
    days = np.round(base + noise).astype(int)
    days = np.clip(days, 0, None)
    df = pd.DataFrame({"year": years, "heatwave_days_seoul": days})
    return df

# 공개 데이터 로더 (실제 URL을 넣어 시도; 실패 시 예시 사용)
@st.cache_data(ttl=60*60)
def load_public_datasets():
    notices = []
    # 시도 1: NOAA Pathfinder — (예시로 파일 URL이 없을 수 있음)
    try:
        # 실제 운영 시에는 여기에 NetCDF 파일 URL을 넣으세요.
        # 예: "https://psl.noaa.gov/thredds/dodsC/Datasets/noaa.oisst.v2.highres/sst.day.mean.nc"
        PATHFINDER_URL = "https://www.ncei.noaa.gov/data/pathfinder-sst/combined/pathfinder-v5.3-daily-mean.nc"  # placeholder
        ds_bytes = download_text(PATHFINDER_URL, max_retries=2)
        # 시도: xarray 열기 (from bytes -> need to write temp file)
        with open("/tmp/pathfinder.nc", "wb") as f:
            f.write(ds_bytes)
        ds = xr.open_dataset("/tmp/pathfinder.nc")
        # 예: 변수명 sst, time, lat, lon
        # 간단히 전 지구 평균 시계열 생성
        da = ds.get("sst", None)
        if da is None:
            raise ValueError("sst variable not found in dataset")
        # compute global monthly mean
        # convert to pandas series
        sst_monthly = da.resample(time="1M").mean(dim="time").mean(dim=["lat", "lon"]).to_series()
        df_sst = sst_monthly.reset_index()
        df_sst.columns = ["date", "sst_global_mean_C"]
        # 제거: 오늘 이후의 데이터 삭제
        df_sst["date"] = pd.to_datetime(df_sst["date"])
        df_sst = df_sst[df_sst["date"] <= TODAY]
        return {"sst": df_sst, "kma_heatwave": load_kma_heatwave_example(), "notice": notices}
    except Exception as e:
        notices.append(f"NOAA Pathfinder 데이터 로드 실패: {str(e)} — 예시 데이터로 대체합니다.")
        # fallback example
        df_sst = load_noaa_pathfinder_example()
        df_sst = df_sst[df_sst["date"] <= TODAY]
        notices.append("대체 데이터는 교육/시연용 예시입니다. (실제 분석 시 원본 데이터를 연결하세요)")
        # 기상청 폭염일수는 포털에서 다운로드 권장 — 대체 예시 로드
        df_kma = load_kma_heatwave_example()
        return {"sst": df_sst, "kma_heatwave": df_kma, "notice": notices}

# --- 사용자 입력 데이터 (프롬프트 기반 내장 예시)
# 요구조건: 앱 실행 중 파일 업로드/텍스트 입력 요구 금지 — 따라서 프롬프트의 설명을 바탕으로 예시 데이터셋 생성
@st.cache_data(ttl=60*60)
def load_user_input_example():
    """
    사용자가 제공한 보고서(문장 기반)를 바탕으로 만든 예시 데이터:
      - 학생 설문: '해수온 상승을 중요한 문제로 인식하는가' (예: 찬성/반대/보통)
      - 교실 영향 조사: '폭염으로 인한 수업 영향' (비율)
      - 지역별 해수면 온도 예시 (동해 연안 2010-2024 월평균)
    실제 수치가 없는 경우 교육용 예시로 내장.
    """
    # 학생 설문 (예시)
    survey = pd.DataFrame({
        "response": ["중요하게 인식함", "보통", "중요하지 않음"],
        "count": [128, 45, 27]
    })

    # 교실 영향 (예시: 여러 항목 비율)
    impacts = pd.DataFrame({
        "impact": ["집중력 저하", "수업 단축/취소", "건강 문제(두통/탈수)", "기타"],
        "percent": [45, 25, 20, 10]
    })

    # 동해 연안(예시) — 년-월 시계열
    months = pd.date_range("2010-01-01", "2024-12-01", freq="MS")
    np.random.seed(2)
    trend = 10 + 0.02 * (np.arange(len(months)))  # 상승 추세 (예시)
    seasonal = 3.0 * np.sin(2 * np.pi * (months.month - 1) / 12)
    noise = np.random.normal(scale=0.3, size=len(months))
    sst_east = trend + seasonal + noise
    df_east = pd.DataFrame({"date": months, "sst_east_C": sst_east})
    df_east = df_east[df_east["date"] <= TODAY]
    return {"survey": survey, "impacts": impacts, "sst_east": df_east}

# Load datasets
with st.spinner("공개 데이터와 예시 데이터를 불러오는 중..."):
    public = load_public_datasets()
    user_input = load_user_input_example()

# 화면 상단: 제목 & 경고(데이터 대체 여부)
st.title("바다의 온도 경고음: 해수온 상승과 교실·폭염 연관성 분석")
st.markdown("**미림마이스터고 — 데이터 기반 리포트 대시보드 (한국어 UI)**")

if public.get("notice"):
    for n in public["notice"]:
        st.warning(n)

# 탭: 공개 데이터 / 사용자 입력 데이터
tab1, tab2 = st.tabs(["공식 공개 데이터 대시보드", "사용자 입력 대시보드 (보고서 기반 예시)"])

# --------------------------
# 탭 1: 공개 데이터 대시보드
# --------------------------
with tab1:
    st.header("공식 공개 데이터 — 해수면 온도 (NOAA 예시) 및 서울 폭염일수 (기상청)")
    col1, col2 = st.columns([2,1])

    df_sst = public["sst"].copy()
    df_kma = public["kma_heatwave"].copy()

    # 데이터 표준화: date, value, group(optional)
    df_sst_std = df_sst.rename(columns={"sst_global_mean_C":"value"})
    df_sst_std["date"] = pd.to_datetime(df_sst_std["date"])
    df_sst_std["group"] = "전지구 평균 SST (예시)"
    df_sst_std = df_sst_std[["date", "value", "group"]].drop_duplicates().reset_index(drop=True)

    # 전처리: 결측/형변환/미래데이터 제거(이미 처리됨)
    df_sst_std = df_sst_std.dropna(subset=["date","value"])

    # 시각화: 시계열 (꺾은선 + 이동평균 선택)
    with col1:
        st.subheader("전지구 해수면 온도(월별) — (예시/교육용)")
        smoothing = st.sidebar.slider("이동평균(개월)", 1, 24, 6, help="시계열 스무딩 기간을 선택하세요.")
        df_plot = df_sst_std.copy()
        df_plot = df_plot.sort_values("date")
        if smoothing > 1:
            df_plot["value_smooth"] = df_plot["value"].rolling(smoothing, min_periods=1, center=True).mean()
            ycol = "value_smooth"
        else:
            ycol = "value"

        fig = px.line(df_plot, x="date", y=ycol, title=f"전지구 월별 해수면 온도 (이동평균={smoothing}개월)",
                      labels={"date":"연월", ycol:"해수면 온도 (°C)"})
        fig.update_layout(legend_title_text=None)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**데이터 표(전처리된)**")
        st.dataframe(df_sst_std.head(200))

        # CSV 다운로드
        csv_buf = io.StringIO()
        df_sst_std.to_csv(csv_buf, index=False)
        st.download_button("전처리된 전지구 해수면 온도 CSV 다운로드", data=csv_buf.getvalue(), file_name="sst_global_processed.csv", mime="text/csv")

    with col2:
        st.subheader("서울 연별 폭염일수 (기상청 예시)")
        st.markdown("폭염일수: 연간 일 최고기온이 33°C 이상인 날의 수 (기상청 정의 기준).")
        df_kma_plot = df_kma.copy()
        fig2 = px.bar(df_kma_plot, x="year", y="heatwave_days_seoul", labels={"year":"연도","heatwave_days_seoul":"폭염일수(일)"},
                      title="서울 연별 폭염일수(예시)")
        st.plotly_chart(fig2, use_container_width=True)
        csv_buf2 = io.StringIO()
        df_kma_plot.to_csv(csv_buf2, index=False)
        st.download_button("서울 폭염일수 CSV 다운로드", data=csv_buf2.getvalue(), file_name="seoul_heatwave_days.csv", mime="text/csv")

    # 간단한 상관(예시): 연도 기반 SST(연간평균) vs 폭염일수
    st.markdown("### 해수면 온도(연평균)와 서울 폭염일수의 간단 비교(예시)")
    try:
        df_sst_year = df_sst_std.copy()
        df_sst_year["year"] = df_sst_year["date"].dt.year
        df_sst_annual = df_sst_year.groupby("year", as_index=False)["value"].mean().rename(columns={"value":"sst_annual_mean"})
        df_kma_merge = pd.merge(df_sst_annual, df_kma, on="year", how="inner")
        if not df_kma_merge.empty:
            fig3 = px.scatter(df_kma_merge, x="sst_annual_mean", y="heatwave_days_seoul",
                              trendline="ols",
                              labels={"sst_annual_mean":"연평균 해수면 온도(°C)", "heatwave_days_seoul":"서울 폭염일수(일)"},
                              title="연평균 해수면 온도 vs 서울 폭염일수 (예시)")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("회귀선이 표시된 산점도 — 상관은 인과를 증명하지 않습니다.")
            st.dataframe(df_kma_merge.head(50))
        else:
            st.info("공개 데이터(연도 기준 병합 결과)가 비어있어 상관 분석을 표시할 수 없습니다.")
    except Exception as e:
        st.error(f"간단 비교 시 오류 발생: {e}")

# --------------------------
# 탭 2: 사용자 입력 데이터 대시보드
# --------------------------
with tab2:
    st.header("사용자 입력(보고서 기반 예시) — 미림마이스터고 설문/교실 영향 분석 (내장 예시)")
    survey = user_input["survey"].copy()
    impacts = user_input["impacts"].copy()
    sst_east = user_input["sst_east"].copy()

    # 설문 결과 파이 차트
    colA, colB = st.columns([1,2])
    with colA:
        st.subheader("학생 설문: 해수온 상승 인식도 (예시)")
        fig_pie = px.pie(survey, names="response", values="count", title="해수온 상승에 대한 학생 인식 분포")
        st.plotly_chart(fig_pie, use_container_width=True)
        csv_buf = io.StringIO()
        survey.to_csv(csv_buf, index=False)
        st.download_button("학생 설문 CSV 다운로드", data=csv_buf.getvalue(), file_name="survey_sample.csv", mime="text/csv")

    with colB:
        st.subheader("교실 영향(복수응답 비율, 예시)")
        fig_bar = px.bar(impacts, x="impact", y="percent", labels={"impact":"영향 항목","percent":"비율(%)"},
                         title="폭염으로 인한 교실 영향 (예시)")
        st.plotly_chart(fig_bar, use_container_width=True)
        csv_buf2 = io.StringIO()
        impacts.to_csv(csv_buf2, index=False)
        st.download_button("교실 영향 CSV 다운로드", data=csv_buf2.getvalue(), file_name="class_impacts_sample.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("동해 연안(예시) 해수면 온도 시계열")
    smoothing2 = st.slider("동해 연안 이동평균(개월)", 1, 24, 3)
    df_plot2 = sst_east.copy().sort_values("date")
    if smoothing2 > 1:
        df_plot2["sst_smooth"] = df_plot2["sst_east_C"].rolling(smoothing2, min_periods=1, center=True).mean()
        ycol2 = "sst_smooth"
    else:
        ycol2 = "sst_east_C"

    fig4 = px.area(df_plot2, x="date", y=ycol2, labels={"date":"연월", ycol2:"해수면 온도 (°C)"},
                   title=f"동해 연안 월별 해수면 온도 (예시, 이동평균={smoothing2}개월)")
    st.plotly_chart(fig4, use_container_width=True)
    csv_buf3 = io.StringIO()
    sst_east.to_csv(csv_buf3, index=False)
    st.download_button("동해 연안 예시 시계열 CSV 다운로드", data=csv_buf3.getvalue(), file_name="sst_east_sample.csv", mime="text/csv")

    st.markdown("### 제언 & 다음 단계 (보고서 텍스트 기반)")
    st.markdown("""
    1. 학교 단위 '기후 데이터 탐사대' 결성 — NOAA / 기상청 데이터 직접 수집·분석 권장.  
    2. 교실 온도 모니터링(간단 센서 설치)과 사진/데이터 수집 후 교육청·학생회에 제출.  
    3. 에너지 절약 캠페인(블라인드 사용 규칙, 전기기기 끄기 등)을 학급 단위로 실천.  
    (위 권장 사항은 보고서에서 제시한 내용을 기반으로 한 실행 권장안입니다.)
    """)

st.markdown("---")
st.caption("참고: 공개 데이터는 원본 소스(예: NOAA Pathfinder/OISST, 기상청)를 연결하여 사용하시길 권장합니다. 본 데모는 연결 실패 시 동작하는 예시 데이터를 포함합니다.")

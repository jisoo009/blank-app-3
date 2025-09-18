# streamlit_app.py
"""
Streamlit 앱: 해수온 대시보드 (한국어 UI)
- 공개 데이터 대시보드: NOAA SST (Pathfinder / OISST) + 기상청 폭염일수(서울)
- 사용자 입력 대시보드: 프롬프트에 제공된 텍스트 기반 설문/요약 데이터를 내장 예시로 시각화
- 추가: 사용자가 제공한 뉴스 기사(URL 목록)를 런타임에 시도해 받아와 간단 요약/메타정보 표시
- 기능:
  - 원격 데이터 불러오기(재시도), 실패 시 예시 데이터 자동 대체(화면 안내)
  - 전처리: 결측값 처리, 형변환, 중복 제거, '오늘(앱 실행일) 이후 데이터 제거'
  - 캐싱: @st.cache_data 사용
  - CSV 다운로드(전처리 표)
  - 한국어 UI (라벨/툴팁/버튼)
- 폰트: /fonts/Pretendard-Bold.ttf 적용 시도 (없으면 자동으로 기본 폰트)
- 참고(사용자 제공 URL):
  - https://www.nocutnews.co.kr/news/6202671?utm_source=naver&utm_medium=article&utm_campaign=20240829100108
  - https://www.newspenguin.com/news/articleView.html?idxno=13978
  - https://www.kmib.co.kr/article/view.asp?arcid=0028634793&code=61121111&cp=nv
  - https://www.weeklyseoul.net/news/articleView.html?idxno=80741
  - https://www.news1.kr/it-science/general-science/5657800
  - https://news.kbs.co.kr/news/pc/view/view.do?ncd=5545888&ref=A
- 외부 기사 중 일부는 CORS/사이트 방어 등으로 런타임에 접근 실패할 수 있습니다. 실패 시 사용자에게 안내하고 계속 실행합니다.
- (코드 주석에 원문 URL를 남김)
"""

import io
import os
import time
from datetime import date
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import xarray as xr
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt

st.set_page_config(page_title="해수온 대시보드 — 미림마이스터고", layout="wide")

# --------------------
# 폰트 적용: 안정적으로 존재 여부 확인 후 적용
# --------------------
FONT_PATH = "/fonts/Pretendard-Bold.ttf"
try:
    import matplotlib.font_manager as fm
    if os.path.exists(FONT_PATH):
        fm.fontManager.addfont(FONT_PATH)
        plt.rcParams['font.family'] = fm.FontProperties(fname=FONT_PATH).get_name()
except Exception:
    # 폰트가 없거나 적용 실패하면 기본 폰트로 계속합니다.
    pass

# helper: 오늘 날짜 (현지 시스템 기준)
TODAY = pd.to_datetime(date.today())

# --------------------
# 네트워크 유틸리티 (재시도 + 캐시)
# --------------------
@st.cache_data(ttl=60*60)
def download_bytes(url: str, max_retries: int = 2, timeout: int = 15) -> bytes:
    last_exc = None
    for i in range(max_retries + 1):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "streamlit-app/1.0"})
            r.raise_for_status()
            return r.content
        except Exception as e:
            last_exc = e
            time.sleep(1 + i)
    raise last_exc

@st.cache_data(ttl=60*60)
def fetch_article_meta(url: str) -> dict:
    """
    시도해서 HTML에서 <title>, meta description, 첫 문단(가능하면) 추출.
    접근 실패 시 'error' 키에 메시지 저장.
    """
    try:
        html = download_bytes(url, max_retries=2, timeout=10).decode('utf-8', errors='replace')
        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string.strip() if soup.title and soup.title.string else "")[:250]
        desc = ""
        # 메타 description
        m = soup.find("meta", attrs={"name": "description"}) or soup.find("meta", attrs={"property": "og:description"})
        if m and m.get("content"):
            desc = m["content"].strip()
        # 본문에서 첫 문단 추출(heuristic)
        p = soup.find("p")
        first_p = p.get_text().strip() if p else ""
        return {"url": url, "title": title, "description": desc, "first_paragraph": first_p}
    except Exception as e:
        return {"url": url, "error": f"다운로드/파싱 실패: {e}"}

# --------------------
# 공개 데이터(예시) 로더: 실제 데이터 연결 시 URL을 교체하세요.
# --------------------
@st.cache_data(ttl=60*60)
def load_noaa_example():
    yrs = pd.date_range("1985-01-01", "2024-12-01", freq="MS")
    np.random.seed(0)
    base = 15 + 0.015 * np.arange(len(yrs))
    seasonal = 1.5 * np.sin(2 * np.pi * (yrs.month - 1) / 12)
    noise = np.random.normal(scale=0.2, size=len(yrs))
    sst = base + seasonal + noise
    df = pd.DataFrame({"date": yrs, "sst_global_mean_C": sst})
    return df

@st.cache_data(ttl=60*60)
def load_kma_example():
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
    """
    실제 환경에서는 NOAA/OISST 등의 NetCDF URL을 여기에 넣어 xarray로 읽어오세요.
    실패 시 예시 데이터로 대체하고 경고를 반환합니다.
    """
    notices = []
    try:
        # 실 운영 시에는 실제 NetCDF/CSV URL을 대체하세요.
        PATHFINDER_URL = "https://www.ncei.noaa.gov/data/pathfinder-sst/combined/pathfinder-v5.3-daily-mean.nc"
        # 시도: (많은 환경에서 직접 접근이 안 될 수 있음)
        ds_bytes = download_bytes(PATHFINDER_URL, max_retries=1, timeout=10)
        with open("/tmp/pathfinder.nc", "wb") as f:
            f.write(ds_bytes)
        ds = xr.open_dataset("/tmp/pathfinder.nc")
        da = ds.get("sst", None)
        if da is None:
            raise ValueError("sst 변수 없음")
        sst_monthly = da.resample(time="1M").mean(dim="time").mean(dim=["lat", "lon"]).to_series()
        df_sst = sst_monthly.reset_index()
        df_sst.columns = ["date", "sst_global_mean_C"]
        df_sst["date"] = pd.to_datetime(df_sst["date"])
        df_sst = df_sst[df_sst["date"] <= TODAY]
        return {"sst": df_sst, "kma_heatwave": load_kma_example(), "notice": notices}
    except Exception as e:
        notices.append(f"공개 데이터 연결 실패: {e}. 예시 데이터로 대체합니다.")
        df_sst = load_noaa_example()
        df_sst = df_sst[df_sst["date"] <= TODAY]
        notices.append("대체 데이터는 교육/시연용 예시입니다. (실제 분석 시 원본 데이터를 연결하세요)")
        return {"sst": df_sst, "kma_heatwave": load_kma_example(), "notice": notices}

# --------------------
# 사용자 입력(프롬프트 기반) 예시 데이터 (앱 실행 시 외부 입력 요구하지 않음)
# --------------------
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
    seasonal = 3.0 * np.sin(2 * np.pi * (months.month - 1) / 12)
    noise = np.random.normal(scale=0.3, size=len(months))
    sst_east = trend + seasonal + noise
    df_east = pd.DataFrame({"date": months, "sst_east_C": sst_east})
    df_east = df_east[df_east["date"] <= TODAY]
    return {"survey": survey, "impacts": impacts, "sst_east": df_east}

# --------------------
# 사용자가 제공한 기사 URL 목록 (원문 링크는 코드 주석 상단에 있음)
# 런타임에 시도해 메타정보(제목/서두)만 가져옴. 접근 불가시 실패 메시지 표기.
# --------------------
NEWS_URLS = [
    "https://www.nocutnews.co.kr/news/6202671?utm_source=naver&utm_medium=article&utm_campaign=20240829100108",
    "https://www.newspenguin.com/news/articleView.html?idxno=13978",
    "https://www.kmib.co.kr/article/view.asp?arcid=0028634793&code=61121111&cp=nv",
    "https://www.weeklyseoul.net/news/articleView.html?idxno=80741",
    "https://www.news1.kr/it-science/general-science/5657800",
    "https://news.kbs.co.kr/news/pc/view/view.do?ncd=5545888&ref=A"
]

# --------------------
# 데이터 로드 (공개 + 사용자 예시)
# --------------------
with st.spinner("데이터 및 기사 메타 정보를 불러오는 중..."):
    public = load_public_datasets()
    user_input = load_user_input_example()

# 기사 메타 시도 (비동기 아님 — 실행 중 바로 시도)
article_metas = []
for u in NEWS_URLS:
    article_metas.append(fetch_article_meta(u))

# --------------------
# UI: 제목/공지
# --------------------
st.title("바다의 온도 경고음: 해수온 상승과 교실·폭염 연관성 분석")
st.markdown("**미림마이스터고 — 데이터 기반 리포트 대시보드 (한국어 UI)**")

# 공개 데이터 로드 실패 알림
if public.get("notice"):
    for n in public["notice"]:
        st.warning(n)

# 탭 분할
tab1, tab2, tab3 = st.tabs(["공식 공개 데이터 대시보드", "사용자 입력 대시보드 (보고서 기반 예시)", "참고 기사 요약"])

# --------------------------
# 탭1: 공개 데이터 대시보드
# --------------------------
with tab1:
    st.header("공식 공개 데이터 — 해수면 온도 (예시) 및 서울 폭염일수 (예시)")
    col1, col2 = st.columns([2, 1])

    df_sst = public["sst"].copy()
    df_kma = public["kma_heatwave"].copy()

    # 표준화
    df_sst_std = df_sst.rename(columns={"sst_global_mean_C": "value"})
    df_sst_std["date"] = pd.to_datetime(df_sst_std["date"])
    df_sst_std["group"] = "전지구 평균 SST (예시)"
    df_sst_std = df_sst_std[["date", "value", "group"]].drop_duplicates().reset_index(drop=True)
    df_sst_std = df_sst_std.dropna(subset=["date", "value"])

    with col1:
        st.subheader("전지구 해수면 온도(월별) — (예시/교육용)")
        smoothing = st.sidebar.slider("이동평균(개월)", 1, 24, 6, help="시계열 스무딩 기간을 선택하세요.")
        df_plot = df_sst_std.sort_values("date").copy()
        if smoothing > 1:
            df_plot["value_smooth"] = df_plot["value"].rolling(smoothing, min_periods=1, center=True).mean()
            ycol = "value_smooth"
        else:
            ycol = "value"

        fig = px.line(df_plot, x="date", y=ycol,
                      title=f"전지구 월별 해수면 온도 (이동평균={smoothing}개월)",
                      labels={"date": "연월", ycol: "해수면 온도 (°C)"})
        fig.update_layout(legend_title_text=None)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**데이터 표(전처리된)**")
        st.dataframe(df_sst_std.head(200))

        csv_buf = io.StringIO()
        df_sst_std.to_csv(csv_buf, index=False)
        st.download_button("전처리된 전지구 해수면 온도 CSV 다운로드",
                           data=csv_buf.getvalue(), file_name="sst_global_processed.csv", mime="text/csv")

    with col2:
        st.subheader("서울 연별 폭염일수 (예시)")
        st.markdown("폭염일수: 연간 일 최고기온이 33°C 이상인 날의 수 (기상청 정의 기준).")
        df_kma_plot = df_kma.copy()
        fig2 = px.bar(df_kma_plot, x="year", y="heatwave_days_seoul",
                      labels={"year": "연도", "heatwave_days_seoul": "폭염일수(일)"},
                      title="서울 연별 폭염일수(예시)")
        st.plotly_chart(fig2, use_container_width=True)
        csv_buf2 = io.StringIO()
        df_kma_plot.to_csv(csv_buf2, index=False)
        st.download_button("서울 폭염일수 CSV 다운로드",
                           data=csv_buf2.getvalue(), file_name="seoul_heatwave_days.csv", mime="text/csv")

    # 연평균 비교 (간단 상관)
    st.markdown("### 해수면 온도(연평균)와 서울 폭염일수의 간단 비교(예시)")
    try:
        df_sst_year = df_sst_std.copy()
        df_sst_year["year"] = df_sst_year["date"].dt.year
        df_sst_annual = df_sst_year.groupby("year", as_index=False)["value"].mean().rename(columns={"value": "sst_annual_mean"})
        df_kma_merge = pd.merge(df_sst_annual, df_kma, on="year", how="inner")
        if not df_kma_merge.empty:
            fig3 = px.scatter(df_kma_merge, x="sst_annual_mean", y="heatwave_days_seoul",
                              trendline="ols",
                              labels={"sst_annual_mean": "연평균 해수면 온도(°C)", "heatwave_days_seoul": "서울 폭염일수(일)"},
                              title="연평균 해수면 온도 vs 서울 폭염일수 (예시)")
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown("회귀선이 표시된 산점도 — 상관은 인과를 증명하지 않습니다.")
            st.dataframe(df_kma_merge.head(50))
        else:
            st.info("공개 데이터(연도 기준 병합 결과)가 비어있어 상관 분석을 표시할 수 없습니다.")
    except Exception as e:
        st.error(f"간단 비교 시 오류 발생: {e}")

# --------------------------
# 탭2: 사용자 입력 데이터 대시보드
# --------------------------
with tab2:
    st.header("사용자 입력(보고서 기반 예시) — 미림마이스터고 설문/교실 영향 분석 (내장 예시)")
    survey = user_input["survey"].copy()
    impacts = user_input["impacts"].copy()
    sst_east = user_input["sst_east"].copy()

    colA, colB = st.columns([1, 2])
    with colA:
        st.subheader("학생 설문: 해수온 상승 인식도 (예시)")
        fig_pie = px.pie(survey, names="response", values="count", title="해수온 상승에 대한 학생 인식 분포")
        st.plotly_chart(fig_pie, use_container_width=True)
        csv_buf = io.StringIO()
        survey.to_csv(csv_buf, index=False)
        st.download_button("학생 설문 CSV 다운로드", data=csv_buf.getvalue(), file_name="survey_sample.csv", mime="text/csv")

    with colB:
        st.subheader("교실 영향(복수응답 비율, 예시)")
        fig_bar = px.bar(impacts, x="impact", y="percent",
                         labels={"impact": "영향 항목", "percent": "비율(%)"},
                         title="폭염으로 인한 교실 영향 (예시)")
        st.plotly_chart(fig_bar, use_container_width=True)
        csv_buf2 = io.StringIO()
        impacts.to_csv(csv_buf2, index=False)
        st.download_button("교실 영향 CSV 다운로드", data=csv_buf2.getvalue(), file_name="class_impacts_sample.csv", mime="text/csv")

    st.markdown("---")
    st.subheader("동해 연안(예시) 해수면 온도 시계열")
    smoothing2 = st.slider("동해 연안 이동평균(개월)", 1, 24, 3)
    df_plot2 = sst_east.sort_values("date").copy()
    if smoothing2 > 1:
        df_plot2["sst_smooth"] = df_plot2["sst_east_C"].rolling(smoothing2, min_periods=1, center=True).mean()
        ycol2 = "sst_smooth"
    else:
        ycol2 = "sst_east_C"

    fig4 = px.area(df_plot2, x="date", y=ycol2, labels={"date": "연월", ycol2: "해수면 온도 (°C)"},
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

# --------------------------
# 탭3: 참고 기사 요약 (사용자 제공 URL)
# --------------------------
with tab3:
    st.header("참고 기사 요약 (제공해주신 URL 기반)")
    st.markdown("다음 기사들을 시도해 자동 수집·요약했습니다. 일부 사이트는 접근이 차단되거나 CORS/로봇 정책으로 인해 내용을 가져오지 못할 수 있습니다. 실패한 항목은 오류 메시지를 표시합니다.")
    for meta in article_metas:
        if meta.get("error"):
            st.error(f"{meta['url']} — {meta['error']}")
        else:
            st.subheader(meta.get("title") or "제목 없음")
            if meta.get("description"):
                st.write(meta["description"])
            elif meta.get("first_paragraph"):
                st.write(meta["first_paragraph"][:800] + ("..." if len(meta.get("first_paragraph",""))>800 else ""))
            else:
                st.info(f"요약 내용을 찾을 수 없습니다. 원문: {meta['url']}")
            st.markdown(f"[원문 바로가기]({meta['url']})")

st.markdown("---")
st.caption("참고: 공개 데이터는 원본 소스(예: NOAA Pathfinder/OISST, 기상청)를 연결하여 사용하시길 권장합니다. 본 데모는 연결 실패 시 동작하는 예시 데이터를 포함합니다.")

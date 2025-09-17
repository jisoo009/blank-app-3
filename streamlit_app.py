import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import plotly.express as px
import requests
from io import StringIO
import datetime

# Pretendard 폰트 적용 (있을 경우만)
try:
    plt.rcParams['font.family'] = fm.FontProperties(
        fname="/fonts/Pretendard-Bold.ttf"
    ).get_name()
except:
    pass

st.set_page_config(page_title="해수온 상승 대시보드", layout="wide")

# ---------------------------
# 1. 공개 데이터 (NOAA)
# ---------------------------
@st.cache_data
def load_noaa_data():
    """
    NOAA Global Ocean Surface Temperature dataset
    출처: https://psl.noaa.gov/data/timeseries/
    """
    url = "https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/ersst.v5.global.ocean.csv"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df = df.rename(columns={"Year": "date", "Value": "value"})
        df["date"] = pd.to_datetime(df["date"], format="%Y")
        today = pd.Timestamp.today().normalize()
        df = df[df["date"] <= today]
    except Exception:
        df = pd.DataFrame({
            "date": pd.date_range("2000-01-01", periods=10, freq="Y"),
            "value": [0.2, 0.25, 0.3, 0.28, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
        })
        st.warning("NOAA 데이터 로드 실패: 예시 데이터를 사용합니다.")
    return df

noaa_df = load_noaa_data()

st.title("🌊 해수온 상승 대시보드")

st.header("1. 공식 공개 데이터: NOAA 해수면 온도")
fig1 = px.line(noaa_df, x="date", y="value", title="전 지구 해수면 온도 (NOAA)")
fig1.update_layout(xaxis_title="연도", yaxis_title="온도 이상 (℃)")
st.plotly_chart(fig1, use_container_width=True)
st.download_button("📥 NOAA 데이터 CSV 다운로드", noaa_df.to_csv(index=False).encode("utf-8"), "noaa_sea_temp.csv", "text/csv")

# ---------------------------
# 2. 사용자 입력 데이터
# ---------------------------
@st.cache_data
def load_user_data():
    # 입력 설명 기반 예시 데이터 (보고서에서 언급된 해수온 + 폭염일수)
    df = pd.DataFrame({
        "date": pd.date_range("1980-01-01", periods=45, freq="Y"),
        "sea_temp": [14.1 + (i*0.03) for i in range(45)],  # 해수온 상승
        "heatwave_days": [3 + int(i*0.4) for i in range(45)]  # 서울 폭염일수 증가
    })
    return df

user_df = load_user_data()

st.header("2. 사용자 입력 데이터: 해수온과 폭염일수 비교")
col1, col2 = st.columns(2)

with col1:
    fig2 = px.line(user_df, x="date", y="sea_temp", title="동해 해수온 추세 (예시)")
    fig2.update_layout(xaxis_title="연도", yaxis_title="평균 해수온 (℃)")
    st.plotly_chart(fig2, use_container_width=True)

with col2:
    fig3 = px.bar(user_df, x="date", y="heatwave_days", title="서울 폭염일수 추세 (예시)")
    fig3.update_layout(xaxis_title="연도", yaxis_title="폭염일수 (일)")
    st.plotly_chart(fig3, use_container_width=True)

st.download_button("📥 사용자 데이터 CSV 다운로드", user_df.to_csv(index=False).encode("utf-8"), "user_climate.csv", "text/csv")

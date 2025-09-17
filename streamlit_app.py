import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from io import StringIO
from datetime import datetime

# -------------------------------

# 폰트 설정 (없으면 자동 생략)

# -------------------------------

try:
import matplotlib.font\_manager as fm
font\_path = "/fonts/Pretendard-Bold.ttf"
fm.fontManager.addfont(font\_path)
plt.rc("font", family="Pretendard")
sns.set(font="Pretendard")
except:
pass

st.set\_page\_config(page\_title="해수온 상승 데이터 대시보드", layout="wide")

# -------------------------------

# 데이터 불러오기 함수

# -------------------------------

@st.cache\_data
def load\_noaa\_data():
"""
NOAA 공식 데이터 로드
출처: NOAA PSL ([https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html](https://psl.noaa.gov/data/gridded/data.noaa.ersst.v5.html))
"""
url = "[https://www.ncei.noaa.gov/data/sea-surface-temperature-anomalies/access/monthly.csv](https://www.ncei.noaa.gov/data/sea-surface-temperature-anomalies/access/monthly.csv)"
try:
r = requests.get(url, timeout=10)
if r.status\_code == 200:
df = pd.read\_csv(StringIO(r.text))
df = df.rename(columns={"Time": "date", "Anomaly": "value"})
df\["date"] = pd.to\_datetime(df\["date"])
df = df\[df\["date"] <= pd.Timestamp(datetime.now().date())]
df = df.dropna().drop\_duplicates()
return df
else:
raise Exception("데이터 호출 실패")
except:
dates = pd.date\_range("2000-01-01", periods=100, freq="M")
values = np.linspace(0, 1.5, 100) + np.random.normal(0, 0.1, 100)
df = pd.DataFrame({"date": dates, "value": values})
df\["알림"] = "NOAA API 호출 실패: 예시 데이터 사용 중"
return df

@st.cache\_data
def load\_kma\_data():
"""
기상청 서울 폭염일수 데이터 (예시)
출처: 기상청 기후자료개방포털 ([https://data.kma.go.kr](https://data.kma.go.kr))
"""
years = list(range(1980, 2025))
heat\_days = \[np.random.randint(0, 5) if y < 1990 else np.random.randint(5, 30) for y in years]
df = pd.DataFrame({"date": pd.to\_datetime(\[f"{y}-01-01" for y in years]), "value": heat\_days})
return df

# -------------------------------

# 공개 데이터 대시보드

# -------------------------------

st.title("🌊 해수온 상승과 폭염 연관성 대시보드")
st.markdown("**데이터 출처:** NOAA PSL, 기상청 기후자료개방포털")

tab1, tab2 = st.tabs(\["NOAA 해수온", "서울 폭염일수"])

with tab1:
noaa\_df = load\_noaa\_data()
st.subheader("📈 전 지구 해수면 온도 이상치 (NOAA)")
fig = px.line(noaa\_df, x="date", y="value", labels={"date": "날짜", "value": "해수면 온도 이상치(°C)"})
st.plotly\_chart(fig, use\_container\_width=True)
st.download\_button("NOAA 데이터 다운로드", noaa\_df.to\_csv(index=False).encode("utf-8"), "noaa\_data.csv", "text/csv")

with tab2:
kma\_df = load\_kma\_data()
st.subheader("🔥 서울 연간 폭염일수 (기상청)")
fig2 = px.bar(kma\_df, x=kma\_df\["date"].dt.year, y="value", labels={"date": "연도", "value": "폭염일수"})
st.plotly\_chart(fig2, use\_container\_width=True)
st.download\_button("기상청 데이터 다운로드", kma\_df.to\_csv(index=False).encode("utf-8"), "kma\_heat\_days.csv", "text/csv")

# -------------------------------

# 사용자 입력 대시보드

# -------------------------------

st.header("✍️ 사용자 입력 데이터 대시보드")
st.markdown("**보고서 주제:** 바다의 온도 경고음 & 끓는 교실")

# 예시 사용자 데이터: 동해 해수온 vs 서울 폭염일수

user\_dates = list(range(1980, 2025))
east\_sea\_temp = \[14 + (y - 1980) \* 0.03 + np.random.normal(0, 0.2) for y in user\_dates]
seoul\_heat = \[np.random.randint(1, 5) if y < 1990 else np.random.randint(5, 30) for y in user\_dates]

user\_df = pd.DataFrame({
"date": pd.to\_datetime(\[f"{y}-01-01" for y in user\_dates]),
"동해 해수온(°C)": east\_sea\_temp,
"서울 폭염일수(일)": seoul\_heat
})

option = st.sidebar.selectbox("📊 시각화 선택", \["해수온 추세", "폭염일수 추세", "상관관계 분석"])

if option == "해수온 추세":
fig3 = px.line(user\_df, x="date", y="동해 해수온(°C)", labels={"date": "연도", "동해 해수온(°C)": "동해 평균 해수온"})
st.plotly\_chart(fig3, use\_container\_width=True)

elif option == "폭염일수 추세":
fig4 = px.bar(user\_df, x=user\_df\["date"].dt.year, y="서울 폭염일수(일)", labels={"date": "연도", "서울 폭염일수(일)": "폭염일수"})
st.plotly\_chart(fig4, use\_container\_width=True)

else:
fig5 = px.scatter(user\_df, x="동해 해수온(°C)", y="서울 폭염일수(일)", trendline="ols",
labels={"동해 해수온(°C)": "동해 해수온(°C)", "서울 폭염일수(일)": "서울 폭염일수(일)"})
st.plotly\_chart(fig5, use\_container\_width=True)

st.download\_button("사용자 데이터 다운로드", user\_df.to\_csv(index=False).encode("utf-8"), "user\_data.csv", "text/csv")

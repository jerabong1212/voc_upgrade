import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# =======================
# 데이터 로드
# =======================
VOC_FILE = "VOC_data.xlsx"
TOMATO_FILE = "environment_tomato.xlsx"
LARVA_FILE = "environment_larva.xlsx"

@st.cache_data
def load_data():
    df_voc = pd.read_excel(VOC_FILE)
    df_tomato = pd.read_excel(TOMATO_FILE)
    df_larva = pd.read_excel(LARVA_FILE)
    return df_voc, df_tomato, df_larva

df_voc, df_tomato, df_larva = load_data()

st.set_page_config(page_title="VOC & 환경 데이터 시각화", layout="wide")
st.title("🌿 VOC 실험 + 환경 데이터 통합 시각화 대시보드")

# =======================
# VOC 분석 섹션
# =======================
st.header("📊 VOC 분석")

voc_compounds = df_voc.columns[2:]  # 첫 2개는 treat, replicate라고 가정
compound = st.selectbox("분석할 VOC 성분을 선택하세요", voc_compounds)

# ANOVA
model = ols(f"{compound} ~ C(treat)", data=df_voc).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
st.subheader("ANOVA 결과")
st.write(anova_table)

# Tukey HSD
tukey = pairwise_tukeyhsd(endog=df_voc[compound], groups=df_voc["treat"], alpha=0.05)
st.subheader("Tukey HSD 결과")
st.write(tukey.summary())

# Screening 기능: 유의한 성분 찾기
st.subheader("유의한 차이가 있는 VOC Screening")
significant_compounds = []
for comp in voc_compounds:
    try:
        mod = ols(f"{comp} ~ C(treat)", data=df_voc).fit()
        anova = sm.stats.anova_lm(mod, typ=2)
        if anova["PR(>F)"][0] < 0.05:
            significant_compounds.append(comp)
    except:
        pass
st.write("유의한 차이가 있었던 성분:", significant_compounds if significant_compounds else "없음")

# VOC boxplot
fig, ax = plt.subplots()
df_voc.boxplot(column=compound, by="treat", ax=ax)
plt.title(f"{compound} 농도 분포")
plt.suptitle("")
plt.xlabel("처리구")
plt.ylabel("농도")
st.pyplot(fig)


# =======================
# 환경 데이터 분석 섹션
# =======================
st.header("🌱 환경 데이터 분석")

env_choice = st.radio("분석할 데이터를 선택하세요", ["토마토", "애벌레"])
df_env = df_tomato if env_choice == "토마토" else df_larva

treats = df_env["treat"].unique()
selected_treat = st.selectbox("treat 선택", treats)

df_filtered = df_env[df_env["treat"] == selected_treat].copy()

# 시계열 그래프
st.subheader(f"{env_choice} Treat {selected_treat} 시계열 데이터")
for col in ["temperature", "humidity", "light"]:
    fig, ax = plt.subplots()
    ax.plot(pd.to_datetime(df_filtered["datetime"]), df_filtered[col], label=col)
    ax.set_xlabel("시간")
    ax.set_ylabel(col)
    ax.legend()
    st.pyplot(fig)

# 광도 분석 (0 이상만 고려)
light_data = df_filtered[df_filtered["light"] > 0]["light"]
light_mean = light_data.mean()
light_std = light_data.std()

# 적산광도 (일별 합계)
df_filtered["date"] = pd.to_datetime(df_filtered["datetime"]).dt.date
daily_dli = df_filtered.groupby("date")["light"].sum()

st.subheader("광 분석 결과")
st.write(f"평균 광도 (양수만): {light_mean:.2f}")
st.write(f"표준편차 (양수만): {light_std:.2f}")

fig, ax = plt.subplots()
daily_dli.plot(kind="bar", ax=ax)
ax.set_title("일일 적산광도(DLI)")
ax.set_xlabel("날짜")
ax.set_ylabel("적산광도")
st.pyplot(fig)

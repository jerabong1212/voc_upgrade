import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.formula.api import ols
import statsmodels.api as sm

# -------------------------
# GitHub 파일 경로 (raw URL 사용)
VOC_PATH = "https://raw.githubusercontent.com/your-username/your-repo/main/VOC_data.xlsx"
TOMATO_PATH = "https://raw.githubusercontent.com/your-username/your-repo/main/environment_tomato.xlsx"
LARVA_PATH = "https://raw.githubusercontent.com/your-username/your-repo/main/environment_larva.xlsx"

# -------------------------
# Streamlit UI
st.sidebar.title("분석 모드 선택")
mode = st.sidebar.radio("Mode", ["VOC 분석", "환경데이터 분석"])

# -------------------------
# VOC 분석
if mode == "VOC 분석":
    df = pd.read_excel(VOC_PATH)
    st.title("VOC 데이터 분석")

    voc_list = [c for c in df.columns if c not in ["treat", "chamber", "line", "progress", "interval"]]
    voc_selected = st.selectbox("분석할 VOC 선택", voc_list)

    # ANOVA & Tukey HSD
    model = ols(f"{voc_selected} ~ C(treat)", data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    st.write("ANOVA 결과", anova_table)

    tukey = pairwise_tukeyhsd(df[voc_selected], df["treat"], alpha=0.05)
    st.write("사후검정 결과", tukey.summary())

    # 유의문자 생성 (간단 버전)
    groups = df["treat"].unique()
    letters = {g:"a" for g in groups}

    # 박스플롯 + 유의문자
    fig, ax = plt.subplots()
    sns.boxplot(x="treat", y=voc_selected, data=df, ax=ax)
    for i, g in enumerate(groups):
        y = df[df["treat"]==g][voc_selected].mean()
        ax.text(i, y+0.2, letters[g], ha="center", fontsize=12, color="red")
    st.pyplot(fig)

    # Screening 기능: 전체 VOC에서 유의한 것만 추출
    sig_voc = []
    for v in voc_list:
        m = ols(f"{v} ~ C(treat)", data=df).fit()
        aov = sm.stats.anova_lm(m, typ=2)
        if aov["PR(>F)"][0] < 0.05:
            sig_voc.append(v)
    st.subheader("📌 유의하게 차이난 VOC 리스트")
    st.write(sig_voc if sig_voc else "없음")

# -------------------------
# 환경데이터 분석
if mode == "환경데이터 분석":
    st.title("환경데이터 분석")
    dataset = st.radio("데이터 선택", ["tomato", "larva"])
    if dataset == "tomato":
        df = pd.read_excel(TOMATO_PATH)
    else:
        df = pd.read_excel(LARVA_PATH)

    treat_selected = st.selectbox("Treat 선택", sorted(df["treat"].unique()))
    df_t = df[df["treat"]==treat_selected].copy()
    df_t["datetime"] = pd.to_datetime(df_t["datetime"])
    df_t["date"] = df_t["datetime"].dt.date

    # 시계열 그래프
    for var in ["temperature", "humidity", "light"]:
        fig, ax = plt.subplots()
        ax.plot(df_t["datetime"], df_t[var])
        ax.set_title(f"{var} (Treat {treat_selected})")
        st.pyplot(fig)

    # 광데이터 요약 (light>0)
    df_light = df_t[df_t["light"]>0].copy()
    summary = df_light.groupby("date")["light"].agg(["mean","std","count"])
    summary["photoperiod"] = summary["count"]
    summary["DLI"] = summary["mean"]*3600*summary["count"]/1e6
    st.subheader("광데이터 요약")
    st.dataframe(summary)

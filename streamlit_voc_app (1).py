# streamlit_voc_app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import statsmodels.api as sm
from statsmodels.formula.api import ols

st.set_page_config(page_title="VOC Analyzer", layout="wide")

# ------------------------------
# 유틸
# ------------------------------
def _ensure_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    return missing

def _to_datetime_if_exists(df: pd.DataFrame, col: str):
    if col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
        except Exception:
            pass
    return df

def _summarize_for_bar(df, x, hue, y="Value", err_type="sem"):
    """
    그룹 평균 및 오차(SD/SEM)를 계산해 막대그래프용 데이터 생성
    """
    g = (
        df.groupby([x, hue], dropna=False)[y]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"mean": "Mean", "std": "SD", "count": "N"})
    )
    g["SEM"] = g["SD"] / np.sqrt(g["N"]).replace(0, np.nan)
    if err_type.lower() == "sd":
        g["ERR"] = g["SD"]
    else:
        g["ERR"] = g["SEM"]
    return g

def _anova_oneway(df, factor, y="Value"):
    """
    일원분산분석. 표본수 2 미만인 그룹이 있으면 제거.
    """
    cnt = df.groupby(factor)[y].count()
    keep = cnt[cnt >= 2].index.tolist()
    df2 = df[df[factor].isin(keep)]
    if len(keep) < 2:
        return None, "유효 그룹이 2개 미만이라 ANOVA를 수행할 수 없습니다."
    try:
        model = ols(f"{y} ~ C({factor})", data=df2).fit()
        table = sm.stats.anova_lm(model, typ=2)
        return table, None
    except Exception as e:
        return None, f"ANOVA 에러: {e}"

# ------------------------------
# 사이드바: 데이터 업로드
# ------------------------------
st.sidebar.title("데이터 업로드")
uploaded = st.sidebar.file_uploader("VOC 데이터 (CSV/XLSX)", type=["csv", "xlsx"])

MODES = ["처리별 VOC 비교", "시간별 VOC 변화", "전체 VOC 스크리닝"]
mode = st.sidebar.radio("분석 모드", MODES, index=0)

if uploaded is None:
    st.info("CSV 또는 XLSX 형식의 **long-format** 데이터를 업로드하세요.\n\n필수 컬럼 예시: `VOC, Value, Treat, Progress` (+선택: `Time`)")
    st.stop()

# 읽기
if uploaded.name.lower().endswith(".csv"):
    df = pd.read_csv(uploaded)
else:
    # 첫 번째 시트만 사용
    df = pd.read_excel(uploaded)

# 공통 전처리
df = df.copy()
df.columns = [c.strip() for c in df.columns]
df = _to_datetime_if_exists(df, "Time")

required_basic = ["VOC", "Value", "Treat", "Progress"]
missing = _ensure_cols(df, required_basic)
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}\n업로드 데이터의 컬럼명을 확인하세요.")
    st.stop()

# ------------------------------
# 1) 처리별 VOC 비교
# ------------------------------
if mode == "처리별 VOC 비교":
    st.header("처리별 VOC 비교")
    # VOC 선택
    vocs = sorted(df["VOC"].dropna().unique().tolist())
    voc = st.selectbox("VOC 선택", vocs)

    sub = df[df["VOC"] == voc].dropna(subset=["Value"])

    # 비교 유형 선택
    compare_type = st.radio(
        "비교 유형",
        ["treatment 내 progress 비교", "progress 내 treatment 비교"],
        help=(
            "• treatment 내 progress 비교: 특정 처리(Treat) 안에서 Progress 집단 간 차이를 비교\n"
            "• progress 내 treatment 비교: 특정 Progress 안에서 처리(Treat) 집단 간 차이를 비교"
        ),
    )

    # 오차 타입
    err_type = st.selectbox("오차막대", ["SEM", "SD"], index=0)

    # --------- treatment 내 progress 비교 ---------
    if compare_type == "treatment 내 progress 비교":
        # 고정할 Treat 선택
        treats = sorted(sub["Treat"].dropna().unique().tolist())
        t_sel = st.selectbox("고정할 처리(Treat) 선택", treats)

        sub2 = sub[sub["Treat"] == t_sel]
        if sub2.empty:
            st.warning("선택한 조건에 데이터가 없습니다.")
        else:
            g = _summarize_for_bar(sub2, x="Progress", hue="Treat", y="Value", err_type=err_type)
            # hue가 모두 동일(Treat 고정)이므로 표시용으로 Progress만 사용
            fig = px.bar(
                g, x="Progress", y="Mean", error_y="ERR",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(yaxis_title=f"Value ({err_type})", bargap=0.25)
            st.plotly_chart(fig, use_container_width=True)

            table, msg = _anova_oneway(sub2, factor="Progress", y="Value")
            st.subheader("ANOVA (Progress 효과 @ Treat 고정)")
            if table is not None:
                st.dataframe(table, use_container_width=True)
            else:
                st.info(msg)

    # --------- progress 내 treatment 비교 ---------
    else:
        # 고정할 Progress 선택
        progs = sorted(sub["Progress"].dropna().unique().tolist())
        p_sel = st.selectbox("고정할 Progress 선택", progs)

        sub2 = sub[sub["Progress"] == p_sel]
        if sub2.empty:
            st.warning("선택한 조건에 데이터가 없습니다.")
        else:
            g = _summarize_for_bar(sub2, x="Treat", hue="Progress", y="Value", err_type=err_type)
            # hue가 모두 동일(Progress 고정)이므로 표시용으로 Treat만 사용
            fig = px.bar(
                g, x="Treat", y="Mean", error_y="ERR",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_layout(yaxis_title=f"Value ({err_type})", bargap=0.25)
            st.plotly_chart(fig, use_container_width=True)

            table, msg = _anova_oneway(sub2, factor="Treat", y="Value")
            st.subheader("ANOVA (Treat 효과 @ Progress 고정)")
            if table is not None:
                st.dataframe(table, use_container_width=True)
            else:
                st.info(msg)

# ------------------------------
# 2) 시간별 VOC 변화
# ------------------------------
elif mode == "시간별 VOC 변화":
    st.header("시간별 VOC 변화")
    vocs = sorted(df["VOC"].dropna().unique().tolist())
    voc = st.selectbox("VOC 선택", vocs)
    sub = df[df["VOC"] == voc].dropna(subset=["Value"])

    color_by = st.selectbox("라인 색상 기준", ["Treat", "Progress"], index=0)
    if "Time" not in sub.columns or sub["Time"].isna().all():
        st.warning("Time 컬럼이 없거나 비어 있어 시간별 분석을 할 수 없습니다.")
    else:
        sub = sub.dropna(subset=["Time"]).sort_values("Time")
        fig = px.line(sub, x="Time", y="Value", color=color_by)
        fig.update_layout(yaxis_title="Value", xaxis_title="Time")
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# 3) 전체 VOC 스크리닝
# ------------------------------
elif mode == "전체 VOC 스크리닝":
    st.header("전체 VOC 스크리닝")
    factor = st.selectbox("검정 요인 선택", ["Treat", "Progress"], index=0)
    results = []
    for voc in sorted(df["VOC"].dropna().unique().tolist()):
        voc_df = df[df["VOC"] == voc].dropna(subset=["Value"])
        table, msg = _anova_oneway(voc_df, factor=factor, y="Value")
        if table is not None:
            try:
                pval = table.loc[f"C({factor})", "PR(>F)"]
                results.append((voc, pval))
            except Exception:
                pass
    if results:
        res_df = pd.DataFrame(results, columns=["VOC", "p-value"]).sort_values("p-value")
        st.dataframe(res_df, use_container_width=True)
    else:
        st.info("유효한 ANOVA 결과가 없습니다. 표본수를 확인하세요.")

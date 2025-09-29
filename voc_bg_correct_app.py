
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="VOC Analyzer (Empty-Background Correction)", layout="wide")

st.title("VOC Analyzer (Treatment×Repetition 배경 보정 옵션 포함)")

# ---- Load data ----
st.sidebar.header("데이터 불러오기")

uploaded = st.sidebar.file_uploader("엑셀(.xlsx) 또는 CSV 업로드", type=["xlsx", "csv"])
default_path = st.sidebar.text_input("또는 로컬 파일 경로", value="voc수합.xlsx")

@st.cache_data(show_spinner=False)
def load_data(file_obj_or_path):
    if file_obj_or_path is None:
        return None
    try:
        if hasattr(file_obj_or_path, "name") or hasattr(file_obj_or_path, "read"):
            name = getattr(file_obj_or_path, "name", "")
            if name.lower().endswith(".csv"):
                return pd.read_csv(file_obj_or_path)
            return pd.read_excel(file_obj_or_path)
        else:
            if str(file_obj_or_path).lower().endswith(".csv"):
                return pd.read_csv(file_obj_or_path, encoding="utf-8")
            return pd.read_excel(file_obj_or_path)
    except Exception as e:
        st.error(f"데이터를 불러오는 중 오류: {e}")
        return None

df = None
if uploaded is not None:
    df = load_data(uploaded)
elif default_path:
    df = load_data(default_path)

if df is None:
    st.info("왼쪽에서 파일을 업로드하거나 경로를 입력하세요.")
    st.stop()

# ---- Column definitions ----
meta_cols = [
    "Treatment","Plant","Larva","Repetition","Sub-repetition",
    "Date","Time","Chamber","Line","Progress","Interval (h)",
    "Temp (℃)","Humid (%)"
]

# Identify VOC columns as numeric columns not in meta
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
voc_cols = [c for c in numeric_cols if c not in meta_cols]

REQUIRED = ["Treatment","Repetition","Progress"]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"필수 컬럼이 없습니다: {missing}")
    st.stop()

# ---- Sidebar filters ----
st.sidebar.header("필터")
treatments = ["전체"] + sorted(df["Treatment"].dropna().unique().tolist())
sel_treat = st.sidebar.selectbox("Treatment", treatments, index=0)
rep_vals = ["전체"] + sorted(df["Repetition"].dropna().unique().tolist())
sel_rep = st.sidebar.selectbox("Repetition", rep_vals, index=0)

sel_voc = st.sidebar.selectbox("VOC 선택", options=voc_cols, index=0)
y_log = st.sidebar.checkbox("Y축 로그 스케일", value=False)
apply_correction = st.sidebar.checkbox("빈챔버 보정 적용하기 (Treatment×Repetition)", value=False)

# ---- Apply filters ----
raw_filtered = df.copy()
if sel_treat != "전체":
    raw_filtered = raw_filtered[raw_filtered["Treatment"] == sel_treat]
if sel_rep != "전체":
    raw_filtered = raw_filtered[raw_filtered["Repetition"] == sel_rep]

# ---- Empty-background correction ----
def background_correct_by_TR(df_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Computes background-corrected VOCs by Treatment×Repetition.
    Returns Treat-only rows with additional *_corrected columns for each VOC.
    """
    if df_filtered.empty:
        return df_filtered.copy()

    # Empty means per (Treatment, Repetition)
    empty_means = (
        df_filtered[df_filtered["Progress"] == "Empty"]
        .groupby(["Treatment", "Repetition"])[voc_cols]
        .mean()
        .reset_index()
    )

    treat_df = df_filtered[df_filtered["Progress"] == "Treat"].copy()
    if treat_df.empty:
        return treat_df

    merged = treat_df.merge(empty_means, on=["Treatment", "Repetition"], suffixes=("", "_emptymean"))
    # Compute corrected columns
    for col in voc_cols:
        if (col in merged.columns) and (col + "_emptymean" in merged.columns):
            merged[col + "_corrected"] = merged[col] - merged[col + "_emptymean"]
    return merged

if apply_correction:
    data_to_use = background_correct_by_TR(raw_filtered)
else:
    data_to_use = raw_filtered.copy()

# ---- Downloads ----
st.sidebar.header("다운로드")
def df_to_csv_bytes(df_):
    return df_.to_csv(index=False).encode("utf-8-sig")

st.sidebar.download_button("⬇️ 현재 화면 데이터 CSV (data_to_use)",
                           data=df_to_csv_bytes(data_to_use),
                           file_name="data_to_use.csv",
                           mime="text/csv",
                           key="dl_data_to_use")

st.sidebar.download_button("⬇️ 필터 적용 원본 CSV (raw_filtered)",
                           data=df_to_csv_bytes(raw_filtered),
                           file_name="raw_filtered.csv",
                           mime="text/csv",
                           key="dl_raw_filtered")

# ---- Preview ----
st.subheader("데이터 미리보기")
st.dataframe(data_to_use.head(20), use_container_width=True)

# ---- Plotting ----
st.subheader("시각화")
col1, col2 = st.columns(2)

# 1) Bar: Treatment별 평균(에러바=표준오차)
with col1:
    if apply_correction:
        # corrected 컬럼 사용
        y_col = f"{sel_voc}_corrected"
        if y_col not in data_to_use.columns:
            st.warning(f"'{sel_voc}'의 보정 컬럼이 없어 원본으로 표시합니다.")
            y_col = sel_voc
    else:
        y_col = sel_voc

    # 집계
    if "Progress" in data_to_use.columns and not apply_correction:
        # 원본일 때만 Progress 구분이 가능(보정은 Treat만 남음)
        group_keys = ["Treatment", "Progress", "Repetition"]
    else:
        group_keys = ["Treatment", "Repetition"]

    grouped = (
        data_to_use.dropna(subset=[y_col])
        .groupby(group_keys)[y_col]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    grouped["err"] = grouped["std"] / grouped["count"].replace(0, np.nan) ** 0.5

    fig = px.bar(
        grouped, x="Treatment", y="mean",
        color="Progress" if ("Progress" in grouped.columns and not apply_correction) else None,
        barmode="group",
        error_y="err",
        labels={"mean": y_col, "Treatment": "처리"},
        title=f"{sel_voc} - 처리별 평균 비교 ({'보정' if apply_correction else '원본'})"
    )
    if y_log:
        fig.update_yaxes(type="log")
    fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig, use_container_width=True)

# 2) Box: Treatment별 분포
with col2:
    if apply_correction:
        y_col = f"{sel_voc}_corrected"
        if y_col not in data_to_use.columns:
            y_col = sel_voc
    else:
        y_col = sel_voc
    box_df = data_to_use.dropna(subset=[y_col])
    fig2 = px.box(
        box_df, x="Treatment", y=y_col,
        points="outliers",
        labels={y_col: y_col, "Treatment": "처리"},
        title=f"{sel_voc} - 처리별 분포 (박스플롯) ({'보정' if apply_correction else '원본'})"
    )
    if y_log:
        fig2.update_yaxes(type="log")
    fig2.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(fig2, use_container_width=True)

st.caption("보정 방식: 같은 Treatment×Repetition에서 Progress=Empty 평균을 Progress=Treat 값에서 차감.")

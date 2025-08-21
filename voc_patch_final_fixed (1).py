import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------- Lazy stats import (only when needed) ----------
def _lazy_import_stats():
    import importlib
    sm = importlib.import_module("statsmodels.api")
    smf = importlib.import_module("statsmodels.formula.api")
    try:
        mc_mod = importlib.import_module("statsmodels.stats.multicomp")
        MultiComparison = getattr(mc_mod, "MultiComparison")
    except Exception:
        MultiComparison = None
    try:
        sp = importlib.import_module("scikit_posthocs")
        HAS_SCPH = True
    except Exception:
        sp = None
        HAS_SCPH = False
    return sm, smf, MultiComparison, sp, HAS_SCPH

st.set_page_config(page_title="VOC 실험 시각화", layout="wide")
st.title("🌿 식물 VOC 실험 결과 시각화")

# =========================================================
# 1) 업로드 + 시트 선택/병합 + 템플릿 다운로드 + 컬럼 자동 매핑
# =========================================================
st.sidebar.header("📁 데이터 불러오기")

@st.cache_data(show_spinner=False)
def _read_any(file_bytes, name):
    ext = str(name).lower().split(".")[-1]
    if ext == "csv":
        return pd.read_csv(io.BytesIO(file_bytes)), None
    xlf = pd.ExcelFile(io.BytesIO(file_bytes))
    return None, xlf.sheet_names

@st.cache_data(show_spinner=False)
def _read_excel_sheet(file_bytes, sheet_name):
    xlf = pd.ExcelFile(io.BytesIO(file_bytes))
    return xlf.parse(sheet_name)

@st.cache_data(show_spinner=False)
def _read_excel_all(file_bytes, sheet_names):
    frames = []
    xlf = pd.ExcelFile(io.BytesIO(file_bytes))
    for s in sheet_names:
        try:
            df = xlf.parse(s)
            df["__Sheet__"] = s
            frames.append(df)
        except Exception:
            pass
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

def _template_bytes():
    template_cols = [
        "Name","Treatment","Start Date","End Date","Chamber","Line",
        "Progress","Interval (h)","Temp (℃)","Humid (%)",
        "Repetition","Sub-repetition",
        "linalool","DMNT","beta-caryophyllene"
    ]
    buf = io.BytesIO()
    pd.DataFrame(columns=template_cols).to_excel(buf, index=False, engine="openpyxl")
    return buf.getvalue()

# 템플릿 다운로드
st.sidebar.download_button(
    "⬇️ 템플릿 엑셀 다운로드",
    data=_template_bytes(),
    file_name="VOC_template.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

uploaded = st.sidebar.file_uploader("VOC 데이터 업로드 (xlsx/xls/csv)", type=["xlsx","xls","csv"])
use_demo = st.sidebar.button("🧪 데모 데이터 불러오기")

df, file_name = None, None
sheet_names = None
file_bytes = None

if uploaded is not None:
    file_bytes = uploaded.getvalue()
    tmp, sheet_names = _read_any(file_bytes, uploaded.name)
    if tmp is not None:  # CSV
        df = tmp
    file_name = uploaded.name

if use_demo and df is None and uploaded is None:
    demo = {
        "Name": ["A"]*18,
        "Treatment": ["control"]*6 + ["herbivory"]*6 + ["threat"]*6,
        "Start Date": pd.to_datetime(["2025-08-01"]*18),
        "End Date": pd.to_datetime(["2025-08-02"]*18),
        "Chamber": ["C1"]*9 + ["C2"]*9,
        "Line": ["L1"]*18,
        "Progress": (["before"]*3 + ["after"]*3)*3,
        "Interval (h)": [-1,0,1, -1,0,1]*3,
        "Temp (℃)": np.random.normal(24, 0.3, 18),
        "Humid (%)": np.random.normal(55, 1.2, 18),
        "Repetition": [1]*18,
        "Sub-repetition": [1,2,3]*6,
        "linalool": np.r_[np.random.normal(5,0.3,6), np.random.normal(7,0.3,6), np.random.normal(9,0.3,6)],
    }
    df = pd.DataFrame(demo)
    file_name = "DEMO"

# 엑셀 다중 시트 지원
if df is None and sheet_names is not None and file_bytes is not None:
    st.sidebar.markdown("**엑셀 시트 구성 감지됨**")
    combine_all = st.sidebar.checkbox("📑 모든 시트 합쳐서 분석", value=False)
    if combine_all:
        df = _read_excel_all(file_bytes, sheet_names)
        st.sidebar.caption("모든 시트를 세로 병합했습니다.")
    else:
        sel_sheet = st.sidebar.selectbox("📑 시트 선택", sheet_names, index=0)
        df = _read_excel_sheet(file_bytes, sel_sheet)

# 파일명/정보 배지
if df is None:
    st.info("왼쪽 사이드바에서 파일을 업로드하거나 **🧪 데모 데이터**로 시작하세요.")
    st.stop()
else:
    st.caption(f"🗂️ 데이터 소스: **{file_name}** | 행: {len(df)}")

# ---------- 표준 컬럼 이름으로 자동 매핑 ----------
CANON = {
    "Name": ["name","sample","시료","샘플","이름"],
    "Treatment": ["treatment","처리","처리구","group","그룹"],
    "Start Date": ["start date","start","시작","시작일"],
    "End Date": ["end date","end","종료","종료일"],
    "Chamber": ["chamber","룸","방","챔버"],
    "Line": ["line","라인","계통","품종"],
    "Progress": ["progress","상태","단계","before/after","stage"],
    "Interval (h)": ["interval (h)","interval","time (h)","time","시간","시간(h)","interval(h)","시각","측정간격"],
    "Temp (℃)": ["temp (℃)","temp","temperature","온도"],
    "Humid (%)": ["humid (%)","humidity","습도"],
    "Repetition": ["repetition","replicate","rep","반복","반복수"],
    "Sub-repetition": ["sub-repetition","technical replicate","subrep","sub_rep","소반복","소반복수"],
}
def _normalize(s):
    return str(s).strip().lower().replace("_"," ").replace("-"," ")

def standardize_columns(df):
    col_map = {}
    for c in df.columns:
        lc = _normalize(c)
        mapped = None
        for canon, aliases in CANON.items():
            if lc == _normalize(canon) or lc in [_normalize(a) for a in aliases]:
                mapped = canon
                break
        if mapped:
            col_map[c] = mapped
    if col_map:
        df = df.rename(columns=col_map)
    return df

df = standardize_columns(df)

# =========================================================
# 2) 애널리틱스 준비: VOC 컬럼 탐지 + 공통 상수
# =========================================================
NAME_COL      = "Name"
TREAT_COL     = "Treatment"
START_COL     = "Start Date"
END_COL       = "End Date"
CHAMBER_COL   = "Chamber"
LINE_COL      = "Line"
PROGRESS_COL  = "Progress"
INTERVAL_COL  = "Interval (h)"
TEMP_COL      = "Temp (℃)"
HUMID_COL     = "Humid (%)"

# 반복/소반복 자동 감지
REP_CANDIDATES    = ["Repetition", "rep", "Rep", "repetition", "반복", "반복수"]
SUBREP_CANDIDATES = ["Sub-repetition", "subrep", "Subrep", "Sub-rep", "sub-repetition", "소반복", "소반복수"]
REP_COL    = next((c for c in REP_CANDIDATES if c in df.columns), None)
SUBREP_COL = next((c for c in SUBREP_CANDIDATES if c in df.columns), None)

VOC_24_CANDIDATES = [
    "(+/-)-trans-nerolidol",
    "(E)-2-hexenal;(Z)-3-hexenal",
    "(S)-citronellol",
    "(Z)-3-hexen-1-ol",
    "(Z)-3-hexenyl acetate",
    "2-phenylethanol",
    "alpha-farnesene",
    "alpha-pinene",
    "benzaldehyde",
    "beta-caryophyllene",
    "beta-pinene",
    "DEN",
    "eucalyptol",
    "indole",
    "lemonol",
    "linalool",
    "methyl jasmonate (20180404ATFtest)",
    "methyl salicylate",
    "nicotine",
    "nitric oxide",
    "ocimene;Limonene;myrcene",
    "Pinenes",
    "toluene",
    "xylenes + ethylbenzene",
]
DISPLAY_MAP = {
    "DEN": "DMNT",
    "DMNT": "DMNT",
    "methyl jasmonate (20180404ATFtest)": "Methyl jasmonate",
    "methyl jasmonate (temporary)": "Methyl jasmonate",
}
def display_name(col):
    return DISPLAY_MAP.get(col, col)

def resolve_voc_columns(df, candidates):
    resolved = []
    for col in candidates:
        if col in df.columns:
            resolved.append(col)
        elif col == "DEN" and "DMNT" in df.columns:
            resolved.append("DMNT")
    meta_cols = set([NAME_COL,TREAT_COL,START_COL,END_COL,CHAMBER_COL,LINE_COL,PROGRESS_COL,INTERVAL_COL,TEMP_COL,HUMID_COL,REP_COL,SUBREP_COL])
    numeric_candidates = [c for c in df.columns if c not in meta_cols and pd.api.types.is_numeric_dtype(df[c])]
    if not resolved and numeric_candidates:
        resolved = numeric_candidates
    return resolved

voc_columns = resolve_voc_columns(df, VOC_24_CANDIDATES)
if not voc_columns:
    st.error("VOC 수치형 컬럼을 찾지 못했습니다. 데이터 헤더를 확인하세요.")
    st.stop()
elif set(voc_columns) != set(VOC_24_CANDIDATES):
    st.info(("데이터에서 감지된 VOC 컬럼: " f"{', '.join([display_name(c) for c in voc_columns])}"))

if INTERVAL_COL in df.columns:
    df[INTERVAL_COL] = pd.to_numeric(df[INTERVAL_COL], errors="coerce")

# =========================================================
# 3) 사이드바 필터/옵션
# =========================================================
st.sidebar.header("🔧 분석 옵션")
chambers = ["전체"] + sorted(df[CHAMBER_COL].dropna().astype(str).unique().tolist()) if CHAMBER_COL in df.columns else ["전체"]
lines    = ["전체"] + sorted(df[LINE_COL].dropna().astype(str).unique().tolist()) if LINE_COL in df.columns else ["전체"]
chamber_sel = st.sidebar.selectbox("🏠 Chamber", chambers, index=0)
line_sel    = st.sidebar.selectbox("🧵 Line", lines, index=0)

treatments = sorted(df[TREAT_COL].dropna().astype(str).unique().tolist()) if TREAT_COL in df.columns else []
treatments_for_ts = ["전체"] + treatments
intervals_all = sorted(df[INTERVAL_COL].dropna().unique().tolist()) if INTERVAL_COL in df.columns else []
reps_all = ["전체"] + sorted(df[REP_COL].dropna().astype(str).unique().tolist()) if REP_COL else ["전체"]
progress_vals_all = sorted(df[PROGRESS_COL].dropna().astype(str).unique().tolist()) if PROGRESS_COL in df.columns else []

rep_sel = st.sidebar.selectbox("🔁 Repetition", reps_all, index=0) if REP_COL else "전체"
progress_sel = st.sidebar.multiselect("🧭 Progress(복수 선택 가능)", progress_vals_all, default=progress_vals_all)

mode = st.sidebar.radio("분석 모드 선택", ["처리별 VOC 비교", "시간별 VOC 변화", "전체 VOC 스크리닝"])

if mode != "전체 VOC 스크리닝":
    selected_voc = st.sidebar.selectbox("📌 VOC 물질 선택", [display_name(c) for c in voc_columns])
    inv_map = {display_name(c): c for c in voc_columns}
    selected_voc_internal = inv_map[selected_voc]
else:
    selected_voc = None
    selected_voc_internal = None

facet_by_chamber = st.sidebar.checkbox("Chamber로 분할 보기", value=False)
facet_by_line    = st.sidebar.checkbox("Line으로 분할 보기", value=False)
err_mode = st.sidebar.radio("오차 기준", ["SD", "SEM"], index=0)
show_subrep_lines = st.sidebar.checkbox("소반복 라인 표시", value=bool(SUBREP_COL)) if SUBREP_COL else False

# =========================================================
# 4) 필터 적용 + 유틸
# =========================================================
def apply_filters(df):
    out = df.copy()
    if CHAMBER_COL in out.columns and chamber_sel != "전체":
        out = out[out[CHAMBER_COL].astype(str) == str(chamber_sel)]
    if LINE_COL in out.columns and line_sel != "전체":
        out = out[out[LINE_COL].astype(str) == str(line_sel)]
    if PROGRESS_COL in out.columns and progress_sel:
        out = out[out[PROGRESS_COL].astype(str).isin(progress_sel)]
    if REP_COL and rep_sel != "전체":
        out = out[out[REP_COL].astype(str) == str(rep_sel)]
    return out

filtered_df = apply_filters(df)

def add_facets(kwargs, data_frame):
    if facet_by_chamber and CHAMBER_COL in data_frame.columns:
        kwargs["facet_col"] = CHAMBER_COL
    if facet_by_line and LINE_COL in data_frame.columns:
        if "facet_col" in kwargs:
            kwargs["facet_row"] = LINE_COL
        else:
            kwargs["facet_col"] = LINE_COL
    return kwargs

def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"

def cld_from_nonsig(groups, pairs_ns):
    groups = list(groups)
    letters = {g: "" for g in groups}
    remaining = set(groups)
    alphabet = list("abcdefghijklmnopqrstuvwxyz")
    li = 0
    while remaining:
        letter = alphabet[li % len(alphabet)]
        bucket = []
        for g in list(remaining):
            ok = True
            for h in bucket:
                pair = tuple(sorted((g, h)))
                if pair not in pairs_ns and g != h:
                    ok = False
                    break
            if ok:
                bucket.append(g)
        for g in bucket:
            letters[g] += letter
            remaining.remove(g)
        li += 1
    for g in groups:
        if letters[g] == "":
            letters[g] = "a"
    return letters

def sem_from_sd(sd, n):
    try:
        return sd / np.sqrt(n) if (sd is not None and n and n > 0) else np.nan
    except Exception:
        return np.nan

def attach_error_col(df_stats, err_mode):
    df_stats = df_stats.copy()
    if "sd" not in df_stats.columns:
        df_stats["sd"] = np.nan
    if "n" not in df_stats.columns:
        df_stats["n"] = np.nan
    if err_mode == "SEM":
        df_stats["err"] = df_stats.apply(lambda r: sem_from_sd(r.get("sd", np.nan), r.get("n", np.nan)), axis=1)
    else:
        df_stats["err"] = df_stats.get("sd", np.nan)
    return df_stats

# =========================================================
# 5) 분석 모드
# =========================================================
if mode in ["처리별 VOC 비교", "시간별 VOC 변화"]:
    if mode == "처리별 VOC 비교":
        chart_type = st.sidebar.radio("차트 유형", ["막대그래프", "박스플롯"], index=0)
        selected_interval = st.sidebar.selectbox("⏱ Interval (h) 선택", ["전체"] + intervals_all)

        show_anova = False
        alpha = 0.05
        include_rep_block = False

        if chart_type == "막대그래프":
            show_anova = st.sidebar.checkbox("ANOVA 분석 표시(막대그래프 전용)", value=False)
            include_rep_block = st.sidebar.checkbox("반복을 블록요인으로 포함", value=bool(REP_COL)) if REP_COL else False
            if show_anova:
                alpha = st.sidebar.number_input("유의수준 α", min_value=0.001, max_value=0.20, value=0.05, step=0.001, format="%.3f")

    else:  # 시간별 VOC 변화
        selected_treatment = st.sidebar.selectbox("🧪 처리구 선택", ["전체"] + treatments)

    
# ----- 처리별 VOC 비교 -----
if mode == "처리별 VOC 비교":
    # 사이드바: 비교 기준 선택
    compare_mode = st.sidebar.radio(
        "비교 기준",
        ["treatment 내 progress 비교", "progress 내 treatment 비교"],
        index=0,
        help="• treatment 내 progress 비교: 각 처리에서 progress 차이를 비교\n• progress 내 treatment 비교: 선택한 progress에서 처리 간 차이를 비교")
    progress_fixed = None
    if compare_mode == "progress 내 treatment 비교" and PROGRESS_COL in filtered_df.columns:
        progress_vals = sorted(map(str, filtered_df[PROGRESS_COL].dropna().unique().tolist()))
        default_index = progress_vals.index("treat") if "treat" in progress_vals else 0
        progress_fixed = st.sidebar.selectbox("기준 Progress 선택", progress_vals, index=default_index)

    chart_type = st.sidebar.radio("차트 유형", ["막대그래프", "박스플롯"], index=0)
    selected_interval = st.sidebar.selectbox("⏱ Interval (h) 선택", ["전체"] + intervals_all)

    # Interval 필터링
    if selected_interval == "전체":
        data_use = filtered_df.copy()
        title_suffix = "모든 시간"
    else:
        data_use = filtered_df[filtered_df[INTERVAL_COL] == selected_interval].copy()
        title_suffix = f"Interval: {selected_interval}h"

    # progress 내 treatment 비교 시 기준 progress로 필터
    title_suffix_extra = ""
    if compare_mode == "progress 내 treatment 비교" and progress_fixed is not None:
        data_use = data_use[data_use[PROGRESS_COL].astype(str) == str(progress_fixed)]
        title_suffix_extra = f" | 고정 Progress: {progress_fixed}"

    y_label = f"{selected_voc} 농도 (ppb)"

    # 색상 구분
    if compare_mode == "treatment 내 progress 비교" and PROGRESS_COL in data_use.columns:
        color_kw = {"color": PROGRESS_COL}
    else:
        color_kw = {}

    # 막대그래프
    if chart_type == "막대그래프":
        group_keys = [TREAT_COL]
        if compare_mode == "treatment 내 progress 비교" and PROGRESS_COL in data_use.columns:
            group_keys.append(PROGRESS_COL)

        grouped = data_use.groupby(group_keys)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index()
        grouped = attach_error_col(grouped, err_mode)

        fig = px.bar(
            grouped, x=TREAT_COL, y="mean",
            error_y="err", barmode="group",
            labels={"mean": y_label, TREAT_COL: "처리"},
            title=f"{selected_voc} - 처리별 평균 비교 ({title_suffix}{title_suffix_extra})",
            **color_kw
        )
        st.plotly_chart(fig, use_container_width=True)

        # ANOVA
        sm, smf, *_ = _lazy_import_stats()
        anova_df = data_use[[TREAT_COL, selected_voc_internal]].dropna()
        if anova_df[TREAT_COL].nunique() >= 2:
            model = smf.ols("Q('{}') ~ C(Q('{}'))".format(selected_voc_internal, TREAT_COL), data=anova_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.subheader("ANOVA 결과")
            st.dataframe(anova_table)

    # 박스플롯
    else:
        fig = px.box(
            data_use, x=TREAT_COL, y=selected_voc_internal,
            labels={selected_voc_internal: y_label, TREAT_COL: "처리"},
            title=f"{selected_voc} - 처리별 분포 (박스플롯) ({title_suffix}{title_suffix_extra})",
            points="outliers",
            **color_kw
        )
        st.plotly_chart(fig, use_container_width=True)
# ----- 시간별 VOC 변화 -----
    else:
        if selected_treatment == "전체":
            data_use = filtered_df.copy()
            title_prefix = "모든 처리"
        else:
            data_use = filtered_df[filtered_df[TREAT_COL].astype(str) == str(selected_treatment)].copy()
            title_prefix = f"{selected_treatment} 처리"

        tick_vals = sorted(df[INTERVAL_COL].dropna().unique().tolist()) if INTERVAL_COL in df.columns else []

        group_keys_display = [INTERVAL_COL]
        if selected_treatment == "전체":
            group_keys_display.append(TREAT_COL)
        if PROGRESS_COL in data_use.columns:
            group_keys_display.append(PROGRESS_COL)
        if CHAMBER_COL in data_use.columns and facet_by_chamber:
            group_keys_display.append(CHAMBER_COL)
        if LINE_COL in data_use.columns and facet_by_line:
            group_keys_display.append(LINE_COL)

        n_rep = data_use[REP_COL].nunique() if REP_COL and REP_COL in data_use.columns else 0

        if SUBREP_COL and SUBREP_COL in data_use.columns:
            per_subrep_ts = (
                data_use.groupby(group_keys_display + (([REP_COL] if REP_COL else []) + [SUBREP_COL]))[selected_voc_internal]
                .mean()
                .reset_index()
            )
        else:
            per_subrep_ts = data_use.copy()

        if n_rep and n_rep >= 2:
            per_rep_ts = per_subrep_ts.groupby(group_keys_display + [REP_COL])[selected_voc_internal].mean().reset_index()
            final = per_rep_ts.groupby(group_keys_display)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
            err_basis = "반복 SD/SEM"
        else:
            final = per_subrep_ts.groupby(group_keys_display)[selected_voc_internal].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
            err_basis = "소반복 SD/SEM"

        final = attach_error_col(final, err_mode)

        fig_kwargs = dict(
            x=INTERVAL_COL,
            y="mean",
            error_y="err",
            markers=True,
            labels={INTERVAL_COL: "Interval (h)", "mean": f"{selected_voc} 평균농도 (ppb)"},
            title=f"{title_prefix} - {selected_voc} 변화 추이 (평균±{err_mode}, 기준: {err_basis})",
        )
        if selected_treatment == "전체":
            fig_kwargs["color"] = TREAT_COL
        elif PROGRESS_COL in data_use.columns:
            fig_kwargs["color"] = PROGRESS_COL
        fig_kwargs = add_facets(fig_kwargs, final)
        fig_voc = px.line(final, **fig_kwargs)
        if tick_vals:
            fig_voc.update_xaxes(tickmode='array', tickvals=tick_vals)
        fig_voc.update_layout(margin=dict(l=10, r=10, t=60, b=10))
        st.plotly_chart(fig_voc, use_container_width=True)

        # ---- 환경변수 그래프를 먼저 그려서, 이후 에러가 나도 이 부분은 보장 ----
        for env_col in [TEMP_COL, HUMID_COL]:
            if env_col not in data_use.columns:
                continue
            if SUBREP_COL and SUBREP_COL in data_use.columns:
                per_subrep_env = (
                    data_use.groupby(group_keys_display + (([REP_COL] if REP_COL else []) + [SUBREP_COL]))[env_col]
                    .mean()
                    .reset_index()
                )
            else:
                per_subrep_env = data_use.copy()

            if n_rep and n_rep >= 2:
                per_rep_env = per_subrep_env.groupby(group_keys_display + ([REP_COL] if REP_COL else []))[env_col].mean().reset_index()
                ts_env = per_rep_env.groupby(group_keys_display)[env_col].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
                err_basis_env = "반복 SD/SEM"
            else:
                ts_env = per_subrep_env.groupby(group_keys_display)[env_col].agg(mean="mean", sd="std", n="count").reset_index().sort_values(INTERVAL_COL)
                err_basis_env = "소반복 SD/SEM"

            ts_env = attach_error_col(ts_env, err_mode)

            ylab = "온도 (°C)" if env_col == TEMP_COL else "상대습도 (%)" if env_col == HUMID_COL else env_col
            fig_kwargs_env = dict(
                x=INTERVAL_COL,
                y="mean",
                error_y="err",
                markers=True,
                labels={INTERVAL_COL: "Interval (h)", "mean": ylab},
                title=f"{title_prefix} - {env_col} 변화 추이 (평균±{err_mode}, 기준: {err_basis_env})",
            )
            if selected_treatment == "전체":
                fig_kwargs_env["color"] = TREAT_COL
            elif PROGRESS_COL in data_use.columns:
                fig_kwargs_env["color"] = PROGRESS_COL
            fig_kwargs_env = add_facets(fig_kwargs_env, ts_env)
            fig_env = px.line(ts_env, **fig_kwargs_env)
            if tick_vals:
                fig_env.update_xaxes(tickmode='array', tickvals=tick_vals)
            fig_env.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig_env, use_container_width=True)

        # ---- 소반복 라인 오버레이: 강력한 예외 처리로 앱 중단 방지 ----
        if show_subrep_lines and SUBREP_COL and SUBREP_COL in data_use.columns:
            try:
                disp_df = per_subrep_ts.rename(columns={selected_voc_internal: "val"})
                # 필수 컬럼 확인 및 정리
                if INTERVAL_COL not in disp_df.columns or "val" not in disp_df.columns:
                    st.info("소반복 라인을 표시할 필수 컬럼(Interval/val)이 없습니다.")
                else:
                    # NaN 제거
                    disp_df = disp_df.dropna(subset=[INTERVAL_COL, "val"])
                    if disp_df.empty:
                        st.info("소반복 라인을 표시할 데이터가 없습니다.")
                    else:
                        fig_kwargs_sub = dict(
                            x=INTERVAL_COL,
                            y="val",
                            hover_data=[c for c in [SUBREP_COL, REP_COL] if c and c in disp_df.columns],
                            opacity=0.35,
                        )
                        if selected_treatment == "전체" and TREAT_COL in disp_df.columns:
                            fig_kwargs_sub["color"] = TREAT_COL
                        elif PROGRESS_COL in disp_df.columns:
                            fig_kwargs_sub["color"] = PROGRESS_COL
                        fig_kwargs_sub = add_facets(fig_kwargs_sub, disp_df)
                        fig_sub = px.line(disp_df, **fig_kwargs_sub)
                        fig_sub.update_traces(line=dict(width=1))
                        st.plotly_chart(fig_sub, use_container_width=True)
            except Exception as e:
                st.warning(f"소반복 라인 오버레이를 건너뜁니다: {e}")

# ----- 전체 VOC 스크리닝 -----
else:
    st.subheader("🔎 전체 VOC 스크리닝 (ANOVA 일괄 분석)")

    selected_interval = st.sidebar.selectbox("⏱ Interval (h) 선택", ["전체"] + intervals_all, key="scr_interval")
    alpha = st.sidebar.number_input("유의수준 α", min_value=0.001, max_value=0.20, value=0.05, step=0.001, format="%.3f", key="scr_alpha")
    do_posthoc = st.sidebar.checkbox("사후검정(Tukey/Duncan) 요약 포함", value=True)
    posthoc_method = st.sidebar.selectbox("사후검정 방법", ["Tukey HSD", "Duncan"], index=0)
    only_sig = st.sidebar.checkbox("유의 VOC만 표시 (p < α)", value=False)
    show_letters_cols = st.sidebar.checkbox("Letters 요약 문자열 포함", value=True)
    include_rep_block_scr = st.sidebar.checkbox("반복을 블록요인으로 포함(스크리닝)", value=bool(REP_COL)) if REP_COL else False

    if selected_interval == "전체":
        data_use = filtered_df.copy()
        title_suffix = "모든 시간"
    else:
        data_use = filtered_df[filtered_df[INTERVAL_COL] == selected_interval].copy()
        title_suffix = f"Interval: {selected_interval}h"

    sm, smf, MultiComparison, sp, HAS_SCPH = _lazy_import_stats()

    results = []
    for voc in voc_columns:
        base_cols = [TREAT_COL, voc]
        if REP_COL: base_cols.append(REP_COL)
        if SUBREP_COL: base_cols.append(SUBREP_COL)
        sub = data_use[base_cols].dropna().copy()
        if sub.empty or sub[TREAT_COL].nunique() < 2:
            continue
        if not all(sub.groupby(TREAT_COL)[voc].count() >= 2):
            continue
        a_df = sub.rename(columns={voc: "y", TREAT_COL: "treat"})
        try:
            if include_rep_block_scr and REP_COL and REP_COL in a_df.columns:
                a_df["rep"] = a_df[REP_COL].astype(str)
                model = smf.ols("y ~ C(treat) + C(rep)", data=a_df).fit()
            else:
                model = smf.ols("y ~ C(treat)", data=a_df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            pval = float(anova_table.loc["C(treat)", "PR(>F)"])
            stars = p_to_stars(pval)
        except Exception:
            pval, stars = np.nan, "ERR"

        results.append({
            "VOC": display_name(voc),
            "p_value": pval,
            "Significance": stars,
        })

    if not results:
        st.info("조건에 맞는 데이터가 없어 스크리닝 결과가 비어있습니다. 필터/Interval/Progress를 확인하세요.")
    else:
        res_df = pd.DataFrame(results).sort_values("p_value", na_position="last")
        if only_sig:
            res_df = res_df[res_df["p_value"] < alpha]
        st.markdown(f"**Interval: {title_suffix}**, α={alpha}")
        st.dataframe(res_df, use_container_width=True)
        st.download_button(
            "⬇️ 스크리닝 결과 CSV",
            data=res_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="voc_screening_results.csv",
            mime="text/csv",
        )

# ---------- 원본 데이터 확인 ----------
with st.expander("🔍 원본 데이터 보기"):
    st.dataframe(df, use_container_width=True)

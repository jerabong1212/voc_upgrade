
# streamlit_voc_app.py
import itertools
import string
import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api as sm
import streamlit as st
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import MultiComparison

st.set_page_config(page_title="VOC Analyzer", layout="wide")

# ------------------------------
# 컬럼 자동 매핑(샘플 양식 반영)
# ------------------------------
META_SYNONYMS = {
    "Treat": ["Treat", "Treatment", "treat", "treatment", "처리"],
    "Progress": ["Progress", "progress", "진행", "Stage", "stage"],
    "Rep": ["Rep", "Repetition", "rep", "반복"],
    "Subrep": ["Subrep", "Sub-repetition", "Sub_repetition", "subrep", "소반복"],
    "Chamber": ["Chamber", "chamber"],
    "Line": ["Line", "line"],
    "Interval": ["Interval (h)", "Interval", "interval", "시간간격", "시간(h)"],
    "Time": ["Time", "Timestamp", "시각"],
    "StartDate": ["Start Date", "Start", "시작일"],
    "EndDate": ["End Date", "End", "종료일"],
    "Temp": ["Temp (℃)", "Temp", "Temperature (℃)", "Temperature"],
    "Humid": ["Humid (%)", "Humidity (%)", "Humid", "Humidity"],
    "VOC": ["VOC", "Analyte", "Compound"],
    "Value": ["Value", "Conc", "Concentration", "ppb", "Intensity"]
}

PREFERRED_NAMES = {
    "Treat": "Treat",
    "Progress": "Progress",
    "Rep": "Rep",
    "Subrep": "Subrep",
    "Chamber": "Chamber",
    "Line": "Line",
    "Interval": "Interval (h)",
    "Time": "Time",
    "StartDate": "Start Date",
    "EndDate": "End Date",
    "Temp": "Temp (℃)",
    "Humid": "Humid (%)",
    "VOC": "VOC",
    "Value": "Value"
}

def _find_first(df_cols, candidates):
    for c in candidates:
        if c in df_cols:
            return c
    return None

def resolve_meta_mapping(df: pd.DataFrame):
    """
    데이터의 실제 컬럼명을 표준 키로 매핑.
    반환: mapping(dict: 표준키 -> 실제컬럼 or None)
    """
    cols = list(df.columns)
    mapping = {}
    for key, cand in META_SYNONYMS.items():
        mapping[key] = _find_first(cols, cand)
    return mapping

def is_long_format(df: pd.DataFrame, mapping):
    # long-format: VOC, Value 컬럼이 명시적으로 존재
    return (mapping.get("VOC") in df.columns) and (mapping.get("Value") in df.columns)

def melt_wide_to_long(df: pd.DataFrame, mapping):
    """
    wide-format -> long-format 변환.
    VOC 후보: 메타 컬럼을 제외한 나머지 숫자형/문자형 모든 열(단, Start/End/Temp/Humid 등 제외).
    """
    # ID 컬럼 후보(존재하는 것만 사용)
    id_keys_order = [
        mapping.get("Treat"), mapping.get("Progress"),
        mapping.get("Rep"), mapping.get("Subrep"),
        mapping.get("Chamber"), mapping.get("Line"),
        mapping.get("Interval"), mapping.get("Time"),
        mapping.get("StartDate"), mapping.get("EndDate"),
        mapping.get("Temp"), mapping.get("Humid"),
    ]
    id_vars = [c for c in id_keys_order if c and c in df.columns]

    # VOC 후보: id_vars 제외한 나머지
    voc_cols = [c for c in df.columns if c not in id_vars]

    # 널리 쓰이는 환경열이 혹시 남아있다면 한 번 더 제외
    env_like = set()
    for key in ["Temp", "Humid", "StartDate", "EndDate", "Interval", "Time"]:
        col = mapping.get(key)
        if col:
            env_like.add(col)
    voc_cols = [c for c in voc_cols if c not in env_like]

    # Melt
    long_df = df.melt(id_vars=id_vars, value_vars=voc_cols, var_name="VOC", value_name="Value")

    # 표준 컬럼명으로 rename 추가(사용자 편의)
    rename_map = {}
    if mapping.get("Treat"): rename_map[mapping["Treat"]] = "Treat"
    if mapping.get("Progress"): rename_map[mapping["Progress"]] = "Progress"
    if mapping.get("Rep"): rename_map[mapping["Rep"]] = "Rep"
    if mapping.get("Subrep"): rename_map[mapping["Subrep"]] = "Subrep"
    if mapping.get("Chamber"): rename_map[mapping["Chamber"]] = "Chamber"
    if mapping.get("Line"): rename_map[mapping["Line"]] = "Line"
    if mapping.get("Interval"): rename_map[mapping["Interval"]] = "Interval (h)"
    if mapping.get("Time"): rename_map[mapping["Time"]] = "Time"
    if mapping.get("StartDate"): rename_map[mapping["StartDate"]] = "Start Date"
    if mapping.get("EndDate"): rename_map[mapping["EndDate"]] = "End Date"
    if mapping.get("Temp"): rename_map[mapping["Temp"]] = "Temp (℃)"
    if mapping.get("Humid"): rename_map[mapping["Humid"]] = "Humid (%)"
    long_df = long_df.rename(columns=rename_map)

    # 값 숫자 변환 시도
    long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
    return long_df

# ------------------------------
# 통계/시각화 유틸
# ------------------------------
def _summarize_for_bar(df, x, y="Value", err_type="sem"):
    g = (
        df.groupby([x], dropna=False)[y]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"mean": "Mean", "std": "SD", "count": "N"})
    )
    g["SEM"] = g["SD"] / np.sqrt(g["N"]).replace(0, np.nan)
    g["ERR"] = g["SD"] if err_type.lower() == "sd" else g["SEM"]
    return g

def _anova_oneway(df, factor, y="Value"):
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

def _tukey_posthoc_with_cld(df, factor, y="Value", alpha=0.05):
    # 그룹당 표본수 2 미만 제외
    cnt = df.groupby(factor)[y].count()
    keep = cnt[cnt >= 2].index.tolist()
    df2 = df[df[factor].isin(keep)].copy()
    levels = sorted(df2[factor].dropna().unique().tolist())
    if len(levels) < 2:
        return pd.DataFrame(), pd.DataFrame(columns=[factor, "letter"])

    mc = MultiComparison(df2[y], df2[factor])
    tukey = mc.tukeyhsd(alpha=alpha)

    # posthoc 결과표
    tbl = tukey.summary()
    headers = tbl.data[0]
    rows = tbl.data[1:]
    posthoc_df = pd.DataFrame(rows, columns=headers)
    for col in ["meandiff", "lower", "upper", "p-adj"]:
        posthoc_df[col] = pd.to_numeric(posthoc_df[col], errors="coerce")
    posthoc_df["reject"] = posthoc_df["reject"].astype(bool)

    # CLD 생성
    groups = list(mc.groupsunique)
    k = len(groups)
    combs = list(itertools.combinations(range(k), 2))
    rejects = list(tukey.reject)
    sig_map = {}
    for (i, j), r in zip(combs, rejects):
        a, b = groups[i], groups[j]
        sig_map[frozenset((a, b))] = bool(r)

    means = df2.groupby(factor)[y].mean().reindex(groups)
    order = list(means.sort_values(ascending=False).index)

    letter_groups = []
    for g in order:
        placed = False
        for s in letter_groups:
            if all(not sig_map.get(frozenset((g, h)), False) for h in s):
                s.add(g)
                placed = True
                break
        if not placed:
            letter_groups.append({g})

    alphabet = list(string.ascii_uppercase)
    letters_map = {lvl: "" for lvl in groups}
    for idx, s in enumerate(letter_groups):
        letter = alphabet[idx] if idx < len(alphabet) else f"L{idx+1}"
        for lvl in s:
            letters_map[lvl] += letter

    cld_df = pd.DataFrame({factor: list(letters_map.keys()), "letter": list(letters_map.values())})
    return posthoc_df, cld_df

def _annotate_letters(fig, df_bar, x_col, y_col="Mean", err_col="ERR", letter_df=None):
    if letter_df is None or letter_df.empty:
        return fig
    m = pd.merge(df_bar[[x_col, y_col, err_col]], letter_df, left_on=x_col, right_on=x_col, how="left")
    if df_bar[y_col].notnull().any():
        y_span = df_bar[y_col].max() - df_bar[y_col].min()
    else:
        y_span = 1.0
    offset = (y_span if y_span > 0 else 1.0) * 0.06
    for _, r in m.iterrows():
        x = r[x_col]
        base = r[y_col] if pd.notnull(r[y_col]) else 0.0
        err = r[err_col] if pd.notnull(r[err_col]) else 0.0
        y = base + err + offset
        text = r.get("letter", "")
        if not isinstance(text, str):
            text = ""
        fig.add_annotation(x=x, y=y, text=text, showarrow=False, font=dict(size=14))
    return fig

# ------------------------------
# 사이드바
# ------------------------------
st.sidebar.title("데이터 업로드")
uploaded = st.sidebar.file_uploader("VOC 데이터 (CSV/XLSX)", type=["csv", "xlsx"])

MODES = ["처리별 VOC 비교", "시간별 VOC 변화", "전체 VOC 스크리닝"]
mode = st.sidebar.radio("분석 모드", MODES, index=0)
alpha = st.sidebar.number_input("유의수준 α (Tukey HSD)", min_value=0.001, max_value=0.3, value=0.05, step=0.005)

if uploaded is None:
    st.info("CSV 또는 XLSX **wide/long 모두 지원**. 샘플 양식(예: Treatment, Repetition, Sub-repetition, Progress, Interval (h), Temp (℃), Humid (%), [VOC...])을 그대로 올려도 자동 인식합니다.\n\nlong-format 필수 컬럼: `VOC, Value, Treat, Progress` (+선택: `Time`)")
    st.stop()

# 데이터 읽기
if uploaded.name.lower().endswith(".csv"):
    df_raw = pd.read_csv(uploaded)
else:
    df_raw = pd.read_excel(uploaded)

df_raw = df_raw.copy()
df_raw.columns = [c.strip() for c in df_raw.columns]

# 자동 매핑 + 포맷 판별
mapping = resolve_meta_mapping(df_raw)

if is_long_format(df_raw, mapping):
    df = df_raw.rename(columns={mapping["VOC"]:"VOC", mapping["Value"]:"Value"})
    # 나머지 메타도 표준화
    if mapping.get("Treat"): df = df.rename(columns={mapping["Treat"]:"Treat"})
    if mapping.get("Progress"): df = df.rename(columns={mapping["Progress"]:"Progress"})
    if mapping.get("Time"): df = df.rename(columns={mapping["Time"]:"Time"})
    if mapping.get("Interval"): df = df.rename(columns={mapping["Interval"]:"Interval (h)"})
else:
    df = melt_wide_to_long(df_raw, mapping)

# 필수 확인
required_basic = ["VOC", "Value", "Treat", "Progress"]
missing = [c for c in required_basic if c not in df.columns]
if missing:
    st.error(f"필수 컬럼이 없습니다(자동 변환 후에도 누락): {missing}\n업로드 데이터의 컬럼명을 확인하거나, 매핑 규칙을 확장해 주세요.")
    st.stop()

# 타입 정리
df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
if "Interval (h)" in df.columns:
    df["Interval (h)"] = pd.to_numeric(df["Interval (h)"], errors="coerce")

# ------------------------------
# 1) 처리별 VOC 비교
# ------------------------------
if mode == "처리별 VOC 비교":
    st.header("처리별 VOC 비교")
    vocs = sorted(df["VOC"].dropna().unique().tolist())
    voc = st.selectbox("VOC 선택", vocs)

    sub = df[df["VOC"] == voc].dropna(subset=["Value"])

    compare_type = st.radio(
        "비교 유형",
        ["treatment 내 progress 비교", "progress 내 treatment 비교"],
        help=(
            "• treatment 내 progress 비교: 특정 처리(Treat) 안에서 Progress 집단 간 차이\n"
            "• progress 내 treatment 비교: 특정 Progress 안에서 처리(Treat) 집단 간 차이"
        ),
    )
    err_type = st.selectbox("오차막대", ["SEM", "SD"], index=0)

    if compare_type == "treatment 내 progress 비교":
        treats = sorted(sub["Treat"].dropna().unique().tolist())
        t_sel = st.selectbox("고정할 처리(Treat) 선택", treats)
        sub2 = sub[sub["Treat"] == t_sel]

        if sub2.empty:
            st.warning("선택한 조건에 데이터가 없습니다.")
        else:
            # 막대 요약
            g = _summarize_for_bar(sub2, x="Progress", y="Value", err_type=err_type)
            fig = px.bar(g, x="Progress", y="Mean", error_y="ERR", color="Progress",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(yaxis_title=f"Value ({err_type})", bargap=0.25, showlegend=False)

            # ANOVA + Tukey + CLD
            anova_tbl, msg = _anova_oneway(sub2, factor="Progress", y="Value")
            st.subheader("ANOVA (Progress 효과 @ Treat 고정)")
            if anova_tbl is not None:
                st.dataframe(anova_tbl, use_container_width=True)
                post, cld = _tukey_posthoc_with_cld(sub2, factor="Progress", y="Value", alpha=alpha)
                st.subheader(f"Tukey HSD (α={alpha})")
                if not post.empty:
                    st.dataframe(post, use_container_width=True)
                else:
                    st.info("사후검정 수행 불가(유효 그룹<2 또는 표본수 부족).")
                fig = _annotate_letters(fig, g, x_col="Progress", y_col="Mean", err_col="ERR", letter_df=cld.rename(columns={"Progress":"Progress"}))
            else:
                st.info(msg)

            st.plotly_chart(fig, use_container_width=True)

    else:  # progress 내 treatment 비교
        progs = sorted(sub["Progress"].dropna().unique().tolist())
        p_sel = st.selectbox("고정할 Progress 선택", progs)
        sub2 = sub[sub["Progress"] == p_sel]

        if sub2.empty:
            st.warning("선택한 조건에 데이터가 없습니다.")
        else:
            g = _summarize_for_bar(sub2, x="Treat", y="Value", err_type=err_type)
            fig = px.bar(g, x="Treat", y="Mean", error_y="ERR", color="Treat",
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(yaxis_title=f"Value ({err_type})", bargap=0.25, showlegend=False)

            anova_tbl, msg = _anova_oneway(sub2, factor="Treat", y="Value")
            st.subheader("ANOVA (Treat 효과 @ Progress 고정)")
            if anova_tbl is not None:
                st.dataframe(anova_tbl, use_container_width=True)
                post, cld = _tukey_posthoc_with_cld(sub2, factor="Treat", y="Value", alpha=alpha)
                st.subheader(f"Tukey HSD (α={alpha})")
                if not post.empty:
                    st.dataframe(post, use_container_width=True)
                else:
                    st.info("사후검정 수행 불가(유효 그룹<2 또는 표본수 부족).")
                fig = _annotate_letters(fig, g, x_col="Treat", y_col="Mean", err_col="ERR", letter_df=cld.rename(columns={"Treat":"Treat"}))
            else:
                st.info(msg)

            st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# 2) 시간별 VOC 변화
# ------------------------------
elif mode == "시간별 VOC 변화":
    st.header("시간별 VOC 변화")
    vocs = sorted(df["VOC"].dropna().unique().tolist())
    voc = st.selectbox("VOC 선택", vocs)
    sub = df[df["VOC"] == voc].dropna(subset=["Value"]).copy()

    # 시간축 후보 결정
    time_col = None
    for c in ["Time", "Interval (h)", "Start Date"]:
        if c in sub.columns and not sub[c].isna().all():
            time_col = c
            break
    if time_col is None:
        st.warning("Time/Interval/Start Date 중 유효한 시간축 컬럼이 없어 시간별 분석을 할 수 없습니다.")
    else:
        if time_col == "Time":
            try:
                sub["Time"] = pd.to_datetime(sub["Time"])
            except Exception:
                pass
        color_by = st.selectbox("라인 색상 기준", ["Treat", "Progress"], index=0)
        sub = sub.dropna(subset=[time_col]).sort_values(time_col)
        fig = px.line(sub, x=time_col, y="Value", color=color_by)
        fig.update_layout(yaxis_title="Value", xaxis_title=time_col)
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
        anova_tbl, msg = _anova_oneway(voc_df, factor=factor, y="Value")
        if anova_tbl is not None and f"C({factor})" in anova_tbl.index:
            pval = anova_tbl.loc[f"C({factor})", "PR(>F)"]
            results.append((voc, pval))
    if results:
        res_df = pd.DataFrame(results, columns=["VOC", "p-value"]).sort_values("p-value")
        st.dataframe(res_df, use_container_width=True)
    else:
        st.info("유효한 ANOVA 결과가 없습니다. 표본수를 확인하세요.")

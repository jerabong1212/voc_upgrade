
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="환경 비교 (Treat, DLI, Photoperiod)", layout="wide")
st.title("🍅🪲 생육환경 비교 — Treat 기반 분석 + DLI + Photoperiod")

# ---------------------------
# 유틸
# ---------------------------
def to_dt(s):
    return pd.to_datetime(s, errors="coerce")

def find_col(cols, keywords, exclude=None):
    exclude = exclude or []
    for c in cols:
        lc = str(c).lower()
        if any(k.lower() in lc for k in keywords) and not any(e.lower() in lc for e in exclude):
            return c
    return None

def normalize_env(df, kind):
    """
    kind: 'tomato' or 'larva'
    표준 컬럼으로 변환:
      Treat, Timestamp, PPFD, Temp_C, RH, SowingDate, TransplantingDate, EggDate, HatchDate
    Treat은 문자열로 보관하여 멀티선택에 유연하게 대응.
    """
    if df is None or df.empty:
        return None

    cols = list(df.columns)

    # 필수(가능한 한 유연 탐지)
    treat_col = find_col(cols, ["treat"]) or "Treat"
    date_col  = find_col(cols, ["date"], exclude=["sowing", "transplant", "egg", "hatch"])
    time_col  = find_col(cols, ["time"])

    # 센서
    light_col = find_col(cols, ["par", "ppfd", "light", "lux"])
    temp_col  = find_col(cols, ["temperature", "temp", "℃"])
    rh_col    = find_col(cols, ["relative humidity", "rh", "humid"])

    # 키 날짜
    sow_col   = find_col(cols, ["sowing"])
    trans_col = find_col(cols, ["transplant"])
    egg_col   = find_col(cols, ["egg"])
    hatch_col = find_col(cols, ["hatch"])

    d = df.copy()

    # Timestamp 결합 (date + time)
    if date_col is not None and time_col is not None:
        d["Timestamp"] = to_dt(d[date_col].astype(str) + " " + d[time_col].astype(str))
    elif date_col is not None:
        d["Timestamp"] = to_dt(d[date_col])
    else:
        d["Timestamp"] = pd.NaT
    d = d.dropna(subset=["Timestamp"]).sort_values("Timestamp")

    # 표준 이름 매핑/캐스팅
    if light_col is not None and light_col in d.columns:
        d["PPFD"] = pd.to_numeric(d[light_col], errors="coerce").clip(lower=0)  # 음수 방지
    if temp_col is not None and temp_col in d.columns:
        d["Temp_C"] = pd.to_numeric(d[temp_col], errors="coerce")
    if rh_col is not None and rh_col in d.columns:
        d["RH"] = pd.to_numeric(d[rh_col], errors="coerce")

    # Treat
    if treat_col in d.columns:
        d["Treat"] = d[treat_col].astype(str)
    else:
        d["Treat"] = "1"

    # 중요 날짜: treat와 함께 이동 (행마다 동일 값일 수 있음 → treat별 유니크 사용)
    for src, dst in [(sow_col, "SowingDate"),
                     (trans_col, "TransplantingDate"),
                     (egg_col, "EggDate"),
                     (hatch_col, "HatchDate")]:
        if src is not None and src in d.columns:
            d[dst] = to_dt(d[src])
        else:
            d[dst] = pd.NaT

    keep = ["Timestamp", "Treat"]
    for c in ["PPFD","Temp_C","RH","SowingDate","TransplantingDate","EggDate","HatchDate"]:
        if c in d.columns: keep.append(c)
    return d[keep]

def compute_photoperiod(df, thr=0.0):
    """
    Photoperiod per day per treat (hours).
    PPFD > thr 구간을 '빛이 켜짐'으로 간주하여 샘플 간 Δt를 적분.
    """
    if df is None or df.empty or "PPFD" not in df:
        return pd.DataFrame(columns=["Treat","date","photoperiod_h"])
    d = df[["Timestamp","PPFD","Treat"]].dropna(subset=["Timestamp"]).copy()
    d = d.sort_values("Timestamp")
    d["is_light"] = d["PPFD"] > thr
    # Δt to next sample (h)
    dt = d["Timestamp"].diff().shift(-1).dt.total_seconds()/3600.0
    med = np.nanmedian(dt[dt>0]) if np.isfinite(np.nanmedian(dt[dt>0])) else 0
    dt = dt.fillna(med).clip(lower=0)
    d["piece"] = np.where(d["is_light"], dt, 0.0)
    d["date"] = d["Timestamp"].dt.date
    return d.groupby(["Treat","date"])["piece"].sum().reset_index().rename(columns={"piece":"photoperiod_h"})

def compute_dli(df):
    """
    Daily Light Integral (mol·m⁻²·day⁻¹) per day per treat.
    DLI = ∑ (PPFD[µmol·m⁻²·s⁻¹] × Δt[s]) / 1e6, over one day.
    Δt는 다음 샘플과의 시간 차(초). PPFD는 음수 클립 0.
    """
    if df is None or df.empty or "PPFD" not in df:
        return pd.DataFrame(columns=["Treat","date","DLI_mol_m2_d"])
    d = df[["Timestamp","PPFD","Treat"]].dropna(subset=["Timestamp"]).copy()
    d = d.sort_values("Timestamp")
    # Δt to next sample (s)
    dt = d["Timestamp"].diff().shift(-1).dt.total_seconds()
    med = np.nanmedian(dt[dt>0]) if np.isfinite(np.nanmedian(dt[dt>0])) else 0
    dt = pd.Series(dt).fillna(med).clip(lower=0).values
    # 적산 (µmol·m⁻²) → mol·m⁻²
    mol = d["PPFD"].clip(lower=0).values * dt / 1e6
    d["mol_piece"] = mol
    d["date"] = d["Timestamp"].dt.date
    return d.groupby(["Treat","date"])["mol_piece"].sum().reset_index().rename(columns={"mol_piece":"DLI_mol_m2_d"})

def add_treat_keylines(fig, data, dataset_kind):
    """
    Treat별 키 날짜(토마토: Sowing/Transplanting, 애벌레: Egg/Hatch)를 세로 라인으로 표시.
    날짜는 treat 그룹 내 유니크 유효값을 사용.
    """
    if fig is None or data is None or data.empty:
        return
    if dataset_kind == "토마토":
        date_cols = ["SowingDate","TransplantingDate"]
    else:
        date_cols = ["EggDate","HatchDate"]

    for tval, sub in data.groupby("Treat"):
        for col in date_cols:
            if col in sub.columns:
                uniq = pd.to_datetime(sub[col].dropna().unique())
                for dt in uniq:
                    fig.add_vline(x=dt, line_dash="dash",
                                  annotation_text=f"{col} (treat {tval})",
                                  annotation_position="top left")

# ---------------------------
# 파일 업로드 & 읽기
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    st.subheader("🍅 environment_tomato.xlsx")
    up_t = st.file_uploader("토마토 파일 업로드", type=["xlsx"], key="t")
with col2:
    st.subheader("🪲 environment_larva.xlsx")
    up_l = st.file_uploader("애벌레 파일 업로드", type=["xlsx"], key="l")

def read_xlsx(up, fallback):
    if up is not None:
        try: return pd.read_excel(up)
        except Exception: return None
    try: return pd.read_excel(fallback)
    except Exception: return None

raw_t = read_xlsx(up_t, "environment_tomato.xlsx")
raw_l = read_xlsx(up_l, "environment_larva.xlsx")

norm_t = normalize_env(raw_t, "tomato") if raw_t is not None else None
norm_l = normalize_env(raw_l, "larva")  if raw_l is not None else None

# ---------------------------
# 옵션
# ---------------------------
st.sidebar.header("⚙️ 옵션")
dataset = st.sidebar.radio("데이터 선택", ["토마토", "애벌레"], index=0)
metric  = st.sidebar.selectbox("지표", [
    "온도(Temp_C)",
    "습도(RH)",
    "광도(양수 평균)",
    "광주기(일일)",
    "일일 적산광도 DLI (mol·m⁻²·day⁻¹)",
], index=2)
viz     = st.sidebar.radio("시각화", ["시계열", "기간 평균 막대그래프"], index=0)
phot_thr = st.sidebar.number_input("광주기 임계값 (PPFD > thr)", min_value=0.0, value=0.0, step=1.0)

data = norm_t if dataset=="토마토" else norm_l
dataset_kind = dataset

if data is None or data.empty:
    st.info(f"{dataset} 데이터가 없거나 컬럼을 인식하지 못했습니다. 파일/헤더를 점검하세요.")
    st.stop()

# 기간 & Treat 필터
min_t, max_t = data["Timestamp"].min(), data["Timestamp"].max()
start_date, end_date = st.sidebar.date_input("기간 선택", (min_t.date(), max_t.date()))
treats = sorted(data["Treat"].dropna().astype(str).unique().tolist())
treat_sel = st.sidebar.multiselect("Treat 선택(복수)", treats, default=treats)

mask = (
    (data["Timestamp"] >= pd.to_datetime(start_date)) &
    (data["Timestamp"] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)) &
    (data["Treat"].astype(str).isin(treat_sel))
)
data = data.loc[mask].copy()

# ---------------------------
# 시각화/계산
# ---------------------------
def unit_of(m):
    if "온도" in m: return "°C"
    if "습도" in m: return "%"
    if "DLI" in m:  return "mol·m⁻²·day⁻¹"
    if "광주기" in m: return "h"
    return ""

if metric in ["온도(Temp_C)", "습도(RH)", "광도(양수 평균)"]:
    if metric == "온도(Temp_C)":
        ycol = "Temp_C"
    elif metric == "습도(RH)":
        ycol = "RH"
    else:
        ycol = "PPFD"

    has_col = (ycol in data.columns)
    if not has_col:
        st.warning(f"{ycol} 컬럼을 찾지 못했습니다.")
    else:
        if viz == "시계열":
            plot_df = data[["Timestamp", ycol, "Treat"]].dropna().rename(columns={ycol:"Value"})
            # 광도 시계열은 0 포함 그래프(관찰 그대로). 평균 계산은 양수 필터에서 별도 수행.
            fig = px.line(plot_df, x="Timestamp", y="Value", color="Treat",
                          labels={"Value": f"{metric} {unit_of(metric)}", "Timestamp":"시간"},
                          title=f"{dataset} — {metric} 시계열")
            add_treat_keylines(fig, data, dataset_kind)
            fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
            st.plotly_chart(fig, use_container_width=True)

        else:
            # 기간 평균 막대: 광도는 양수(>0)만 사용
            if metric == "광도(양수 평균)":
                summary = data.copy()
                summary = summary[summary["PPFD"] > 0]
                mean_df = summary.groupby("Treat")["PPFD"].mean().reset_index().rename(columns={"PPFD":"Mean"})
            else:
                mean_df = data.groupby("Treat")[ycol].mean().reset_index().rename(columns={ycol:"Mean"})

            if mean_df.empty:
                st.info("평균을 계산할 데이터가 부족합니다.")
            else:
                fig = px.bar(mean_df, x="Treat", y="Mean", text="Mean",
                             labels={"Treat":"Treat", "Mean":f"{metric} 평균 {unit_of(metric)}"},
                             title=f"{dataset} — {metric} 기간 평균 (treat 비교)")
                fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
                st.plotly_chart(fig, use_container_width=True)

elif metric == "광주기(일일)":
    ph = compute_photoperiod(data, thr=phot_thr)
    if ph.empty:
        st.info("광주기 계산에 필요한 PPFD/시간 정보가 충분하지 않습니다.")
    else:
        ph = ph[ph["Treat"].isin(treat_sel)]
        if viz == "시계열":
            fig = px.line(ph, x="date", y="photoperiod_h", color="Treat",
                          labels={"date":"날짜", "photoperiod_h":"광주기 (h)"},
                          title=f"{dataset} — 일일 광주기 시계열 (treat 비교)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            sum_df = ph.groupby("Treat")["photoperiod_h"].mean().reset_index().rename(columns={"photoperiod_h":"Mean"})
            fig = px.bar(sum_df, x="Treat", y="Mean", text="Mean",
                         labels={"Treat":"Treat", "Mean":"평균 광주기 (h)"},
                         title=f"{dataset} — 평균 광주기 (treat 비교)")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        # 다운로드
        st.download_button("⬇️ 일일 광주기(Photoperiod) CSV",
                           data=ph.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"photoperiod_{dataset}.csv",
                           mime="text/csv")

else:  # DLI
    dli = compute_dli(data)
    if dli.empty:
        st.info("DLI 계산에 필요한 PPFD/시간 정보가 충분하지 않습니다.")
    else:
        dli = dli[dli["Treat"].isin(treat_sel)]
        if viz == "시계열":
            fig = px.line(dli, x="date", y="DLI_mol_m2_d", color="Treat",
                          labels={"date":"날짜", "DLI_mol_m2_d":"DLI (mol·m⁻²·day⁻¹)"},
                          title=f"{dataset} — DLI 시계열 (treat 비교)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            sum_df = dli.groupby("Treat")["DLI_mol_m2_d"].mean().reset_index().rename(columns={"DLI_mol_m2_d":"Mean"})
            fig = px.bar(sum_df, x="Treat", y="Mean", text="Mean",
                         labels={"Treat":"Treat", "Mean":"평균 DLI (mol·m⁻²·day⁻¹)"},
                         title=f"{dataset} — 평균 DLI (treat 비교)")
            fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
            st.plotly_chart(fig, use_container_width=True)

        # 다운로드
        st.download_button("⬇️ 일일 적산광도(DLI) CSV",
                           data=dli.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"dli_{dataset}.csv",
                           mime="text/csv")

# ---------------------------
# 미리보기
# ---------------------------
with st.expander("🔍 정규화 데이터 미리보기"):
    st.dataframe(data.head(30), use_container_width=True)

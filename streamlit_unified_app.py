
import streamlit as st
import pandas as pd
import numpy as np
import io
import requests
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula_api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Optional: Duncan (only if scikit_posthocs exists)
try:
    import scikit_posthocs as sp  # noqa: F401
    HAS_SCPH = True
except Exception:
    HAS_SCPH = False

st.set_page_config(page_title='VOC + Environment Dashboard', layout='wide')
st.title('VOC + Environment Dashboard')

# ======================================================
# Config: Local-first; optional GitHub raw fallback
# ======================================================
LOCAL_VOC = 'VOC_data.xlsx'
LOCAL_TOMATO = 'environment_tomato.xlsx'
LOCAL_LARVA = 'environment_larva.xlsx'

# If local files are missing, optional GitHub raw URLs (edit here if you deploy to cloud)
GITHUB_VOC = ''
GITHUB_TOMATO = ''
GITHUB_LARVA = ''

@st.cache_data
def read_local_or_github(local_path: str, raw_url: str):
    p = Path(local_path)
    if p.exists():
        return pd.read_excel(p)
    if raw_url:
        r = requests.get(raw_url, timeout=30)
        r.raise_for_status()
        return pd.read_excel(io.BytesIO(r.content))
    # Nothing available
    return None

@st.cache_data
def load_all():
    voc = read_local_or_github(LOCAL_VOC, GITHUB_VOC)
    tomato = read_local_or_github(LOCAL_TOMATO, GITHUB_TOMATO)
    larva = read_local_or_github(LOCAL_LARVA, GITHUB_LARVA)
    return voc, tomato, larva

df_voc, df_tomato, df_larva = load_all()

# ======================================================
# Utilities
# ======================================================
def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

def ensure_dt(s):
    return pd.to_datetime(s, errors='coerce')

def infer_columns(df):
    # Guess common columns: datetime, temp, humid, light, treat, key dates
    if df is None:
        return (None,)*9
    lc = {c.lower(): c for c in df.columns}

    def pick(cands):
        for key, orig in lc.items():
            for c in cands:
                if c in key:
                    return orig
        return None

    dt_col = pick(['datetime','timestamp','time','date'])
    temp = pick(['temp','temperature'])
    humid = pick(['humid','humidity','rh'])
    light = pick(['ppfd','light','par'])
    treat = pick(['treat','treatment'])

    # Optional key dates
    sow = pick(['sowing'])
    transplant = pick(['transplant'])
    egg = pick(['egg'])
    hatch = pick(['hatch'])

    return dt_col, temp, humid, light, treat, sow, transplant, egg, hatch

def safe_numeric(s):
    return pd.to_numeric(s, errors='coerce')

def integrate_photoperiod(df, ts_col, ppfd_col, thr=0.0):
    # Photoperiod per day (hours): integrate time where PPFD > thr
    if df is None or ts_col not in df or ppfd_col not in df:
        return pd.DataFrame(columns=['date','photoperiod_h'])
    d = df[[ts_col, ppfd_col]].dropna().copy()
    d[ts_col] = ensure_dt(d[ts_col])
    d = d.sort_values(ts_col)
    dt_h = d[ts_col].diff().shift(-1).dt.total_seconds() / 3600.0
    med = np.nanmedian(dt_h[dt_h>0]) if np.isfinite(np.nanmedian(dt_h[dt_h>0])) else 0
    dt_h = dt_h.fillna(med).clip(lower=0)
    is_light = safe_numeric(d[ppfd_col]).fillna(0) > thr
    d['piece'] = np.where(is_light, dt_h, 0.0)
    d['date'] = d[ts_col].dt.date
    return d.groupby('date')['piece'].sum().reset_index().rename(columns={'piece':'photoperiod_h'})

def integrate_dli(df, ts_col, ppfd_col):
    # DLI (mol m^-2 day^-1) = sum(PPFD[umol m^-2 s^-1] * dt[s]) / 1e6
    if df is None or ts_col not in df or ppfd_col not in df:
        return pd.DataFrame(columns=['date','DLI_mol_m2_d'])
    d = df[[ts_col, ppfd_col]].dropna().copy()
    d[ts_col] = ensure_dt(d[ts_col])
    d = d.sort_values(ts_col)
    dt_s = d[ts_col].diff().shift(-1).dt.total_seconds()
    med = np.nanmedian(dt_s[dt_s>0]) if np.isfinite(np.nanmedian(dt_s[dt_s>0])) else 0
    dt_s = pd.Series(dt_s).fillna(med).clip(lower=0).values
    ppfd = safe_numeric(d[ppfd_col]).fillna(0).clip(lower=0).values
    mol_piece = ppfd * dt_s / 1e6
    d['mol_piece'] = mol_piece
    d['date'] = d[ts_col].dt.date
    return d.groupby('date')['mol_piece'].sum().reset_index().rename(columns={'mol_piece':'DLI_mol_m2_d'})

def compact_letter_display_from_tukey(tukey_res, groups_sorted):
    # Generate CLD letters from Tukey result (greedy algorithm)
    df = pd.DataFrame(data=tukey_res._results_table.data[1:], columns=tukey_res._results_table.data[0])
    sig = {(row['group1'], row['group2']): bool(row['reject']) for _, row in df.iterrows()}
    groups = list(groups_sorted)

    letters = {g: "" for g in groups}
    current_letter = "a"

    def conflicts(g, letter):
        for h in groups:
            if h == g: 
                continue
            if letter in letters[h]:
                pair = (g, h) if (g, h) in sig else (h, g)
                if sig.get(pair, False):
                    return True
        return False

    for g in groups:
        placed = False
        for letter in sorted(set("".join(letters.values()))):
            if not conflicts(g, letter):
                letters[g] += letter
                placed = True
                break
        if not placed:
            letters[g] += current_letter
            current_letter = chr(ord(current_letter) + 1)

    return letters

# ======================================================
# Sidebar mode
# ======================================================
mode = st.sidebar.radio('분석 모드', ['VOC 분석', '환경 데이터'], index=0)

# ======================================================
# VOC Analysis
# ======================================================
if mode == 'VOC 분석':
    st.header('VOC 분석')
    if df_voc is None or df_voc.empty:
        st.warning('VOC_data.xlsx 를 찾지 못했습니다. 동일 폴더에 파일을 두거나, 상단 변수에 GitHub raw URL을 지정하세요.')
        st.stop()

    # Identify treatment & numeric VOCs
    candidates_meta = set([
        'treat','treatment','chamber','line','progress','interval','interval (h)',
        'start date','end date','name','progress','rep','replicate'
    ])
    treat_col = None
    for c in df_voc.columns:
        if c.lower() in ('treat','treatment'):
            treat_col = c
            break
    voc_cols = [c for c in df_voc.columns if (c.lower() not in candidates_meta) and pd.api.types.is_numeric_dtype(df_voc[c])]

    if treat_col is None:
        st.warning("VOC 데이터에 'treat' 컬럼이 필요합니다.")
        st.stop()

    col1, col2 = st.columns([1,1])
    with col1:
        compound = st.selectbox('분석할 VOC', voc_cols, index=0 if voc_cols else None)
    with col2:
        run_posthoc = st.selectbox('사후검정', ['Tukey HSD', 'Duncan (scikit-posthocs)'], index=0)

    if compound:
        data = pd.DataFrame({
            'treat': df_voc[treat_col].astype(str),
            'y': pd.to_numeric(df_voc[compound], errors='coerce')
        }).dropna(subset=['y'])

        st.subheader(f'{compound} — 처리별 분포')
        fig, ax = plt.subplots(figsize=(7,4))
        sns.boxplot(x='treat', y='y', data=data, ax=ax, color='#E0E0E0')
        sns.stripplot(x='treat', y='y', data=data, ax=ax, color='black', size=4, jitter=True)
        ax.set_xlabel('처리'); ax.set_ylabel('농도 (ppb)')
        st.pyplot(fig)

        # ANOVA
        if data['treat'].nunique() >= 2:
            model = ols('y ~ C(treat)', data=data).fit()
            aov = sm.stats.anova_lm(model, typ=2)
            st.subheader('ANOVA')
            st.dataframe(aov)
            p = float(aov['PR(>F)'][0]); st.write(f'p = {p:.4g} ({stars(p)})')

            # Posthoc
            st.subheader('사후검정 결과')
            means = data.groupby('treat')['y'].mean().sort_values(ascending=False)
            if run_posthoc.startswith('Tukey'):
                tukey = pairwise_tukeyhsd(endog=data['y'], groups=data['treat'], alpha=0.05)
                st.text(str(tukey))
                letters = compact_letter_display_from_tukey(tukey, means.index.tolist())
            else:
                if not HAS_SCPH:
                    st.info('scikit-posthocs가 설치되어 있지 않습니다. requirements.txt에 scikit-posthocs 추가 후 다시 시도하세요.')
                    letters = None
                else:
                    import scikit_posthocs as sp
                    duncan = sp.posthoc_duncan(data, val_col='y', group_col='treat', p_adjust=None)
                    st.dataframe(duncan)
                    # Build CLD letters from Duncan matrix
                    sig = {}
                    treats = duncan.index.tolist()
                    for i, g1 in enumerate(treats):
                        for j, g2 in enumerate(treats):
                            if j <= i: 
                                continue
                            sig[(g1,g2)] = bool(duncan.loc[g1, g2] < 0.05)
                    # Greedy letters
                    groups = means.index.tolist()
                    letters = {g: "" for g in groups}
                    current_letter = "a"
                    def conflicts(g, letter):
                        for h in groups:
                            if h==g: continue
                            if letter in letters[h]:
                                pair = (g,h) if (g,h) in sig else (h,g)
                                if sig.get(pair, False):
                                    return True
                        return False
                    for g in groups:
                        placed=False
                        for letter in sorted(set("".join(letters.values()))):
                            if not conflicts(g, letter):
                                letters[g]+=letter; placed=True; break
                        if not placed:
                            letters[g]+=current_letter
                            current_letter = chr(ord(current_letter)+1)

            # Annotate CLD letters on the boxplot
            if letters:
                x_labels = list(data['treat'].unique())
                for i, tr in enumerate(x_labels):
                    y_pos = data.loc[data['treat']==tr, 'y'].max()
                    ax.text(i, y_pos, letters.get(tr,''), ha='center', va='bottom', fontsize=12, color='crimson')
                st.pyplot(fig)

        # Screening across all VOCs
        st.subheader('전체 VOC 스크리닝 (ANOVA p<0.05)')
        sig_rows = []
        for col in voc_cols:
            tmp = pd.DataFrame({
                'treat': df_voc[treat_col].astype(str),
                'y': pd.to_numeric(df_voc[col], errors='coerce')
            }).dropna(subset=['y'])
            if tmp['treat'].nunique() >= 2 and len(tmp) > 2:
                try:
                    m = ols('y ~ C(treat)', data=tmp).fit()
                    a = sm.stats.anova_lm(m, typ=2)
                    pv = float(a['PR(>F)'][0])
                    if pv < 0.05:
                        sig_rows.append({'VOC': col, 'p_value': pv, 'signif': stars(pv)})
                except Exception:
                    pass
        if sig_rows:
            st.dataframe(pd.DataFrame(sig_rows).sort_values('p_value'))
        else:
            st.info('유의한 VOC가 없습니다.')

# ======================================================
# Environment Analysis
# ======================================================
else:
    st.header('환경 데이터 분석')
    dataset = st.radio('데이터셋', ['토마토','애벌레'], horizontal=True, index=0)
    df_env = df_tomato if dataset=='토마토' else df_larva

    if df_env is None or df_env.empty:
        st.warning('환경 데이터 파일을 찾지 못했습니다. 동일 폴더에 두거나 상단 변수에 GitHub raw URL을 지정하세요.')
        st.stop()

    dt_col, t_col, h_col, l_col, treat_col, sow_col, trans_col, egg_col, hatch_col = infer_columns(df_env)
    if not treat_col or not dt_col:
        st.warning('환경 데이터에는 최소한 treat와 시간 컬럼이 필요합니다.')
        st.stop()

    d = df_env.copy()
    d[dt_col] = ensure_dt(d[dt_col])
    if t_col in d: d[t_col] = safe_numeric(d[t_col])
    if h_col in d: d[h_col] = safe_numeric(d[h_col])
    if l_col in d: d[l_col] = safe_numeric(d[l_col]).clip(lower=0)

    treats = sorted(d[treat_col].dropna().astype(str).unique().tolist())
    sel = st.multiselect('treat 선택(복수)', treats, default=treats)

    # Period filter
    min_ts, max_ts = d[dt_col].min(), d[dt_col].max()
    start_date, end_date = st.date_input('기간', (min_ts.date(), max_ts.date()))

    mask = (d[dt_col] >= pd.to_datetime(start_date)) & (d[dt_col] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    d = d.loc[mask & d[treat_col].astype(str).isin(sel)].copy()

    # Photoperiod threshold
    thr = st.slider('Photoperiod 임계값 (PPFD > thr)', 0.0, 100.0, 0.0, 1.0)

    # Time series overlays
    st.subheader('시계열 (treat 오버레이)')
    nrows = sum([c is not None for c in [t_col, h_col, l_col]])
    if nrows == 0:
        st.info('표시할 지표가 없습니다.')
    else:
        fig, axes = plt.subplots(nrows, 1, figsize=(11, 3.2*nrows), sharex=True)
        if nrows == 1:
            axes = [axes]
        idx = 0
        for col, ylab in [(t_col,'온도 (C)'), (h_col,'습도 (%)'), (l_col,'광도 (umol m^-2 s^-1)')]:
            if col is None: 
                continue
            for tr in sel:
                sub = d[d[treat_col].astype(str)==tr]
                axes[idx].plot(sub[dt_col], sub[col], label=f'treat {tr}')
            axes[idx].set_ylabel(ylab); axes[idx].legend(loc='upper right')
            idx += 1
        axes[-1].set_xlabel('시간')
        st.pyplot(fig)

    # Light stats (PPFD>0 only)
    if l_col in d:
        st.subheader('광 요약 (PPFD>0)')
        pos = d[d[l_col] > 0]
        mean_by = pos.groupby(treat_col)[l_col].mean().rename('mean').reset_index()
        std_by  = pos.groupby(treat_col)[l_col].std().rename('std').reset_index()
        st.dataframe(pd.merge(mean_by, std_by, on=treat_col, how='outer'))

        # Photoperiod & DLI per treat/day
        st.subheader('일일 Photoperiod(h) & DLI(mol m^-2 day^-1)')
        ph_frames, dli_frames = [], []
        for tr, g in d.groupby(treat_col):
            ph = integrate_photoperiod(g, dt_col, l_col, thr=thr); ph['Treat']=tr; ph_frames.append(ph)
            dl = integrate_dli(g, dt_col, l_col);                 dl['Treat']=tr; dli_frames.append(dl)
        ph_df = pd.concat(ph_frames, axis=0) if ph_frames else pd.DataFrame(columns=['date','photoperiod_h','Treat'])
        dl_df = pd.concat(dli_frames, axis=0) if dli_frames else pd.DataFrame(columns=['date','DLI_mol_m2_d','Treat'])

        col_a, col_b = st.columns(2)
        with col_a:
            if not ph_df.empty:
                fig, ax = plt.subplots(figsize=(10,4))
                for tr, g in ph_df.groupby('Treat'):
                    ax.plot(g['date'], g['photoperiod_h'], marker='o', label=f'treat {tr}')
                ax.set_title('Photoperiod'); ax.set_ylabel('h'); ax.legend()
                st.pyplot(fig)
        with col_b:
            if not dl_df.empty:
                fig, ax = plt.subplots(figsize=(10,4))
                for tr, g in dl_df.groupby('Treat'):
                    ax.plot(g['date'], g['DLI_mol_m2_d'], marker='o', label=f'treat {tr}')
                ax.set_title('DLI'); ax.set_ylabel('mol m^-2 day^-1'); ax.legend()
                st.pyplot(fig)

        # Period means
        rows = []
        for tr in sel:
            sub = d[d[treat_col].astype(str)==tr]
            row = {'Treat': tr}
            if t_col in sub: row['Temp_mean']  = float(np.nanmean(sub[t_col]))
            if h_col in sub: row['Humid_mean'] = float(np.nanmean(sub[h_col]))
            if l_col in sub: row['Light_mean_pos'] = float(np.nanmean(sub.loc[sub[l_col]>0, l_col]))
            if not ph_df.empty: row['Photoperiod_mean'] = float(ph_df.loc[ph_df['Treat']==tr, 'photoperiod_h'].mean())
            if not dl_df.empty: row['DLI_mean']        = float(dl_df.loc[dl_df['Treat']==tr, 'DLI_mol_m2_d'].mean())
            rows.append(row)
        if rows:
            st.subheader('기간 평균 비교 (treat)')
            st.dataframe(pd.DataFrame(rows))

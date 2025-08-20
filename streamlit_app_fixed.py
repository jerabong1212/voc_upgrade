
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title='VOC & 환경 데이터 통합 대시보드', layout='wide')
st.title('🌿 VOC + 환경 데이터 통합 시각화')

VOC_FILE = 'VOC_data.xlsx'
TOMATO_FILE = 'environment_tomato.xlsx'
LARVA_FILE = 'environment_larva.xlsx'

# =======================
# 공용 유틸
# =======================
@st.cache_data
def load_local_excels(voc_path, tomato_path, larva_path):
    df_voc = pd.read_excel(voc_path)
    df_tomato = pd.read_excel(tomato_path)
    df_larva = pd.read_excel(larva_path)
    return df_voc, df_tomato, df_larva

def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

def infer_env_columns(df):
    # 유연한 컬럼 탐지 (datetime, temperature/temp, humidity/rh, light, treat)
    cols = {c.lower(): c for c in df.columns}
    def pick(keys):
        for k in keys:
            # exact/contains
            for lc, orig in cols.items():
                if k in lc:
                    return orig
        return None

    dt_col  = pick(['datetime', 'timestamp', 'time', 'date'])
    temp    = pick(['temperature', 'temp'])
    humid   = pick(['humidity', 'humid', 'rh'])
    light   = pick(['ppfd', 'light', 'par'])
    treat   = pick(['treat'])
    return dt_col, temp, humid, light, treat

def ensure_ts(s):
    return pd.to_datetime(s, errors='coerce')

def integrate_photoperiod(df, ts_col, ppfd_col, thr=0.0):
    # 일일 광주기(h) 계산: PPFD>thr 구간의 시간 적분
    if ts_col is None or ppfd_col is None or ts_col not in df or ppfd_col not in df:
        return pd.DataFrame(columns=['date','photoperiod_h'])
    d = df[[ts_col, ppfd_col]].dropna().copy()
    d[ts_col] = ensure_ts(d[ts_col])
    d = d.sort_values(ts_col)
    dt = d[ts_col].diff().shift(-1).dt.total_seconds() / 3600.0  # 시간
    med = np.nanmedian(dt[dt > 0]) if np.isfinite(np.nanmedian(dt[dt > 0])) else 0
    dt = dt.fillna(med).clip(lower=0)
    is_light = d[ppfd_col].astype(float) > thr
    d['piece'] = np.where(is_light, dt, 0.0)
    d['date'] = d[ts_col].dt.date
    return d.groupby('date')['piece'].sum().reset_index().rename(columns={'piece':'photoperiod_h'})

def integrate_dli(df, ts_col, ppfd_col):
    # 일일 적산광도(mol m^-2 day^-1) = sum(PPFD[umol m^-2 s^-1] * dt[s]) / 1e6
    if ts_col is None or ppfd_col is None or ts_col not in df or ppfd_col not in df:
        return pd.DataFrame(columns=['date','DLI_mol_m2_d'])
    d = df[[ts_col, ppfd_col]].dropna().copy()
    d[ts_col] = ensure_ts(d[ts_col])
    d = d.sort_values(ts_col)
    dt = d[ts_col].diff().shift(-1).dt.total_seconds()  # 초
    med = np.nanmedian(dt[dt > 0]) if np.isfinite(np.nanmedian(dt[dt > 0])) else 0
    dt = pd.Series(dt).fillna(med).clip(lower=0).values
    ppfd = pd.to_numeric(d[ppfd_col], errors='coerce').fillna(0).clip(lower=0).values
    mol_piece = ppfd * dt / 1e6
    d['mol_piece'] = mol_piece
    d['date'] = d[ts_col].dt.date
    return d.groupby('date')['mol_piece'].sum().reset_index().rename(columns={'mol_piece':'DLI_mol_m2_d'})

# =======================
# 데이터 로드
# =======================
df_voc, df_tomato, df_larva = load_local_excels(VOC_FILE, TOMATO_FILE, LARVA_FILE)

# =======================
# 사이드바 모드
# =======================
mode = st.sidebar.radio('모드를 선택하세요', ['VOC 분석', '환경 데이터'])

# =======================
# VOC 분석
# =======================
if mode == 'VOC 분석':
    st.header('📊 VOC 분석')
    # 메타 컬럼 후보
    meta_candidates = set([
        'treat','treatment','chamber','line','progress','interval','interval (h)',
        'start date','end date','name','progress','rep','replicate'
    ])
    # 숫자형 VOC 컬럼만 추림
    voc_cols = [c for c in df_voc.columns if (c.lower() not in meta_candidates) and pd.api.types.is_numeric_dtype(df_voc[c])]
    treat_col = 'treat' if 'treat' in df_voc.columns else 'Treatment' if 'Treatment' in df_voc.columns else None

    if not treat_col:
        st.warning('VOC 데이터에 treat 컬럼이 필요합니다.')
    else:
        compound = st.selectbox('분석할 VOC', voc_cols, index=0 if voc_cols else None)
        if compound:
            tmp = df_voc[[treat_col]].copy()
            tmp['y'] = pd.to_numeric(df_voc[compound], errors='coerce')
            tmp = tmp.dropna(subset=['y'])
            if tmp[treat_col].nunique() < 2:
                st.info('treat 수준이 2개 이상이어야 ANOVA가 가능합니다.')
            else:
                model = ols('y ~ C(%s)' % treat_col, data=tmp).fit()  # 변수명 안전
                aov = sm.stats.anova_lm(model, typ=2)
                st.subheader('ANOVA 결과')
                st.dataframe(aov)

                # 유의표기
                p = aov['PR(>F)'][0]
                st.write(f'ANOVA p = {p:.4g} ({stars(p)})')

                # Tukey
                st.subheader('Tukey HSD')
                tukey = pairwise_tukeyhsd(endog=tmp['y'], groups=tmp[treat_col], alpha=0.05)
                st.text(str(tukey))

                # 박스플롯
                fig, ax = plt.subplots(figsize=(6,4))
                tmp.boxplot(column='y', by=treat_col, ax=ax)
                ax.set_title(f'{compound} — 처리별 분포')
                ax.set_xlabel('처리')
                ax.set_ylabel('농도 (ppb)')
                plt.suptitle('')
                st.pyplot(fig)

        # 스크리닝
        st.subheader('🔥 전체 VOC 스크리닝 (ANOVA p<0.05)')
        sig = []
        for col in voc_cols:
            ttmp = df_voc[[treat_col]].copy()
            ttmp['y'] = pd.to_numeric(df_voc[col], errors='coerce')
            ttmp = ttmp.dropna(subset=['y'])
            if ttmp[treat_col].nunique() >= 2 and len(ttmp) > 0:
                try:
                    m = ols('y ~ C(%s)' % treat_col, data=ttmp).fit()
                    a = sm.stats.anova_lm(m, typ=2)
                    if a['PR(>F)'][0] < 0.05:
                        sig.append((col, a['PR(>F)'][0]))
                except Exception:
                    pass
        if sig:
            sig_df = pd.DataFrame(sig, columns=['VOC','p_value']).sort_values('p_value')
            sig_df['signif'] = sig_df['p_value'].apply(stars)
            st.dataframe(sig_df)
        else:
            st.info('유의한 VOC가 없습니다.')

# =======================
# 환경 데이터
# =======================
else:
    st.header('🌱 환경 데이터 분석')
    dataset = st.radio('데이터셋 선택', ['토마토','애벌레'], horizontal=True)
    df_env = df_tomato if dataset=='토마토' else df_larva

    # 컬럼 인식
    dt_col, t_col, h_col, l_col, treat_col = infer_env_columns(df_env)
    if not treat_col or not dt_col:
        st.warning('환경 데이터에 treat / datetime 컬럼이 필요합니다.')
        st.stop()

    # 형변환
    d = df_env.copy()
    d[dt_col] = ensure_ts(d[dt_col])
    if t_col in d: d[t_col] = pd.to_numeric(d[t_col], errors='coerce')
    if h_col in d: d[h_col] = pd.to_numeric(d[h_col], errors='coerce')
    if l_col in d: d[l_col] = pd.to_numeric(d[l_col], errors='coerce').clip(lower=0)

    treats = sorted(d[treat_col].dropna().astype(str).unique().tolist())
    sel = st.multiselect('treat 선택(복수)', treats, default=treats)

    # 기간
    min_ts, max_ts = d[dt_col].min(), d[dt_col].max()
    if pd.isna(min_ts) or pd.isna(max_ts):
        st.warning('유효한 시간 데이터가 없습니다.')
        st.stop()
    start_date, end_date = st.date_input('기간 선택', (min_ts.date(), max_ts.date()))

    mask = (d[dt_col] >= pd.to_datetime(start_date)) & (d[dt_col] <= pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    d = d.loc[mask & d[treat_col].astype(str).isin(sel)].copy()

    # 시계열
    st.subheader('시계열 (treat 오버레이)')
    nrows = sum([c is not None for c in [t_col, h_col, l_col]])
    if nrows == 0:
        st.info('표시할 지표(온도/습도/광도)가 없습니다.')
    else:
        fig, axes = plt.subplots(nrows, 1, figsize=(10, 3.2*nrows), sharex=True)
        if nrows == 1:
            axes = [axes]
        idx = 0
        for col, ylab in [(t_col, '온도 (°C)'), (h_col, '습도 (%)'), (l_col, '광도 (umol m^-2 s^-1)')]:
            if col is None: continue
            for tr in sel:
                sub = d[d[treat_col].astype(str)==tr]
                axes[idx].plot(sub[dt_col], sub[col], label=f'treat {tr}')
            axes[idx].set_ylabel(ylab); axes[idx].legend(loc='upper right')
            idx += 1
        axes[-1].set_xlabel('시간')
        st.pyplot(fig)

    # 광 요약: mean/std는 PPFD>0만
    if l_col in d:
        st.subheader('광 요약 (PPFD>0 기준)')
        pos = d[d[l_col] > 0]
        mean_by_treat = pos.groupby(treat_col)[l_col].mean().rename('mean').reset_index()
        std_by_treat  = pos.groupby(treat_col)[l_col].std().rename('std').reset_index()
        light_stats = pd.merge(mean_by_treat, std_by_treat, on=treat_col, how='outer')
        st.dataframe(light_stats)

        # Photoperiod & DLI (treat별 일일)
        st.subheader('광주기(일일, h) & DLI(mol m^-2 day^-1)')
        ph_list = []; dli_list = []
        for tr, g in d.groupby(treat_col):
            ph = integrate_photoperiod(g, dt_col, l_col, thr=0.0); ph['Treat'] = tr; ph_list.append(ph)
            dl = integrate_dli(g, dt_col, l_col);                 dl['Treat'] = tr; dli_list.append(dl)
        ph_df = pd.concat(ph_list, axis=0) if ph_list else pd.DataFrame(columns=['date','photoperiod_h','Treat'])
        dl_df = pd.concat(dli_list, axis=0) if dli_list else pd.DataFrame(columns=['date','DLI_mol_m2_d','Treat'])

        if not ph_df.empty:
            fig, ax = plt.subplots(figsize=(9,4))
            for tr, g in ph_df.groupby('Treat'):
                ax.plot(g['date'], g['photoperiod_h'], marker='o', label=f'treat {tr}')
            ax.set_title('일일 광주기'); ax.set_ylabel('h'); ax.legend(); st.pyplot(fig)

        if not dl_df.empty:
            fig, ax = plt.subplots(figsize=(9,4))
            for tr, g in dl_df.groupby('Treat'):
                ax.plot(g['date'], g['DLI_mol_m2_d'], marker='o', label=f'treat {tr}')
            ax.set_title('일일 DLI'); ax.set_ylabel('mol m^-2 day^-1'); ax.legend(); st.pyplot(fig)

        # 기간 평균 막대
        st.subheader('기간 평균 비교')
        rows = []
        for tr in sel:
            sub = d[d[treat_col].astype(str)==tr]
            row = {'Treat': tr}
            if t_col in sub: row['Temp_mean'] = np.nanmean(sub[t_col])
            if h_col in sub: row['Humid_mean'] = np.nanmean(sub[h_col])
            if l_col in sub: row['Light_mean_pos'] = np.nanmean(sub.loc[sub[l_col]>0, l_col])
            if not dl_df.empty: row['DLI_mean'] = dl_df[dl_df['Treat']==tr]['DLI_mol_m2_d'].mean()
            if not ph_df.empty: row['Photoperiod_mean'] = ph_df[ph_df['Treat']==tr]['photoperiod_h'].mean()
            rows.append(row)
        if rows:
            sum_df = pd.DataFrame(rows)
            st.dataframe(sum_df)

            # 막대 예시: Light_mean_pos / DLI_mean
            fig, axes = plt.subplots(1, 2, figsize=(12,4))
            if 'Light_mean_pos' in sum_df:
                axes[0].bar(sum_df['Treat'].astype(str), sum_df['Light_mean_pos']); axes[0].set_title('평균 광도(PPFD>0)'); axes[0].set_ylabel('umol m^-2 s^-1')
            if 'DLI_mean' in sum_df:
                axes[1].bar(sum_df['Treat'].astype(str), sum_df['DLI_mean']); axes[1].set_title('평균 DLI'); axes[1].set_ylabel('mol m^-2 day^-1')
            st.pyplot(fig)
    else:
        st.info('광도 컬럼을 찾지 못했습니다.')

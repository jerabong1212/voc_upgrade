import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# GitHub raw file paths
VOC_PATH = "https://raw.githubusercontent.com/your-username/your-repo/main/VOC_data.xlsx"
TOMATO_PATH = "https://raw.githubusercontent.com/your-username/your-repo/main/environment_tomato.xlsx"
LARVA_PATH = "https://raw.githubusercontent.com/your-username/your-repo/main/environment_larva.xlsx"

st.title("VOC & Environment Data Analysis")

# Load Data
@st.cache_data
def load_data():
    df_voc = pd.read_excel(VOC_PATH)
    df_tomato = pd.read_excel(TOMATO_PATH)
    df_larva = pd.read_excel(LARVA_PATH)
    return df_voc, df_tomato, df_larva

df_voc, df_tomato, df_larva = load_data()

menu = st.sidebar.radio("Select Analysis", ["VOC Analysis", "Tomato Environment", "Larva Environment"])

if menu == "VOC Analysis":
    st.header("VOC Data Analysis")

    compounds = [c for c in df_voc.columns if c not in ["treat", "chamber", "line", "progress", "interval"]]
    selected = st.selectbox("Select compound", compounds)

    # ANOVA
    df_long = df_voc.melt(id_vars=["treat"], value_vars=[selected], var_name="compound", value_name="y")
    model = smf.ols("y ~ C(treat)", data=df_long).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    st.write("ANOVA Table", anova_table)

    # Tukey HSD
    tukey = pairwise_tukeyhsd(df_long["y"], df_long["treat"])
    st.write(tukey.summary())

    # Plot
    fig, ax = plt.subplots()
    sns.boxplot(x="treat", y="y", data=df_long, ax=ax)
    sns.stripplot(x="treat", y="y", data=df_long, color="black", size=4, jitter=True, ax=ax)
    ax.set_title(f"{selected} by Treat")
    st.pyplot(fig)

elif menu == "Tomato Environment":
    st.header("Tomato Environment Data")

    treat = st.selectbox("Select treat", df_tomato["treat"].unique())
    subset = df_tomato[df_tomato["treat"] == treat]

    # Time series
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    subset.plot(x="datetime", y="temperature", ax=axes[0], title="Temperature")
    subset.plot(x="datetime", y="humidity", ax=axes[1], title="Humidity")
    subset.plot(x="datetime", y="light", ax=axes[2], title="Light Intensity")
    st.pyplot(fig)

    # Stats summary
    light_on = subset[subset["light"] > 0]
    avg_light = light_on["light"].mean()
    std_light = light_on["light"].std()
    daily_dli = light_on.groupby(light_on["datetime"].dt.date)["light"].sum()
    st.write(f"Mean light (on only): {avg_light:.2f}")
    st.write(f"Std light (on only): {std_light:.2f}")
    st.write("Daily DLI", daily_dli)

elif menu == "Larva Environment":
    st.header("Larva Environment Data")

    treat = st.selectbox("Select treat", df_larva["treat"].unique())
    subset = df_larva[df_larva["treat"] == treat]

    # Time series
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
    subset.plot(x="datetime", y="temperature", ax=axes[0], title="Temperature")
    subset.plot(x="datetime", y="humidity", ax=axes[1], title="Humidity")
    subset.plot(x="datetime", y="light", ax=axes[2], title="Light Intensity")
    st.pyplot(fig)

    # Stats summary
    light_on = subset[subset["light"] > 0]
    avg_light = light_on["light"].mean()
    std_light = light_on["light"].std()
    daily_dli = light_on.groupby(light_on["datetime"].dt.date)["light"].sum()
    st.write(f"Mean light (on only): {avg_light:.2f}")
    st.write(f"Std light (on only): {std_light:.2f}")
    st.write("Daily DLI", daily_dli)

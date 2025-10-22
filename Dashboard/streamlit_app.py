
"""
Streamlit Dashboard (streamlit_app.py)
Compact grid-based layout for visualizations.
"""

import sys
import os
from pathlib import Path

# Add "code" folder to Python path
sys.path.append(str(Path(__file__).resolve().parents[1] / "code"))

from eda_report_notebook import analyze_and_report, generate_plot_description

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

from eda_report_notebook import analyze_and_report, generate_plot_description

@st.cache_data
def load_dataframe(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        try:
            return pd.read_excel(uploaded_file)
        except Exception:
            uploaded_file.seek(0)
            text = uploaded_file.read().decode('utf-8', errors='ignore')
            for sep in [',',';','\t','|']:
                try:
                    df = pd.read_csv(io.StringIO(text), sep=sep)
                    if df.shape[1] > 1:
                        return df
                except Exception:
                    continue
    raise ValueError("Unable to parse uploaded file.")

def kpi_card(title, value):
    st.markdown(f"**{title}**")
    st.markdown(f"<h3 style='margin:0'>{value}</h3>", unsafe_allow_html=True)

def fallback_qa(question, df):
    q = question.lower()
    if "rows" in q or "shape" in q:
        return f"The dataset has {df.shape[0]} rows and {df.shape[1]} columns."
    if "columns" in q or "fields" in q:
        return "Columns: " + ", ".join(list(df.columns[:20])) + (", ..." if df.shape[1] > 20 else "")
    if "missing" in q or "null" in q:
        miss = df.isna().sum().sort_values(ascending=False).head(10)
        return "Top missing counts:\n" + miss.to_string()
    if "top" in q and "value" in q:
        for c in df.select_dtypes(include=['object','category']).columns:
            top = df[c].value_counts().head(5)
            return f"Top values in `{c}`:\n" + top.to_string()
    return "Sorry — try asking 'How many rows?' or 'Which column has missing values?'"

st.set_page_config(page_title="EDAI Dashboard",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("<h1>📊 EDAI Dashboard</h1>", unsafe_allow_html=True)

uploaded_file = st.sidebar.file_uploader("Upload CSV / Excel", type=["csv","xlsx","xls"])
show_preview = st.sidebar.checkbox("Show raw data preview", value=True)

df = None
if uploaded_file:
    try:
        df = load_dataframe(uploaded_file)
        st.success(f"Loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")
    except Exception as e:
        st.error(f"Failed to load: {e}")

if df is None:
    st.info("👆 Upload a file to begin analysis.")
    st.stop()

# KPIs
c1, c2, c3, c4 = st.columns(4)
with c1: kpi_card("Rows", f"{df.shape[0]:,}")
with c2: kpi_card("Columns", f"{df.shape[1]:,}")
with c3: kpi_card("Missing values", f"{int(df.isna().sum().sum()):,}")
with c4: kpi_card("Memory", f"{df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

tabs = st.tabs(["Overview","Distributions","Correlations","Time Series","Top Categories","EDA Report","Q&A Chatbot"])

with tabs[0]:
    st.header("Overview")
    if show_preview:
        st.dataframe(df.head(200))
    dtypes = pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str)})
    st.table(dtypes)

with tabs[1]:
    st.header("Distributions & Insights")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    if num_cols:
        chosen = st.selectbox("Numeric column", options=num_cols, key="numcol")
        if chosen:
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(3,2))
                sns.histplot(df[chosen].dropna(), kde=True, ax=ax, color="skyblue")
                ax.set_title(f"Distribution of {chosen}", fontsize=10)
                ax.tick_params(labelsize=8)
                st.pyplot(fig, use_container_width=False)
                st.caption(generate_plot_description(df[[chosen]], "hist"))
            with c2:
                fig2, ax2 = plt.subplots(figsize=(3,2))
                sns.boxplot(x=df[chosen], ax=ax2, color="lightcoral")
                ax2.set_title(f"Boxplot of {chosen}", fontsize=10)
                ax2.tick_params(labelsize=8)
                st.pyplot(fig2, use_container_width=False)
                st.caption(generate_plot_description(df[[chosen]], "box"))

    if num_cols and len(num_cols) > 1:
        st.markdown("#### Scatterplot Insights")
        colx, coly = st.columns(2)
        with colx:
            xcol = st.selectbox("X-axis", options=num_cols, key="sx")
        with coly:
            ycol = st.selectbox("Y-axis", options=num_cols, key="sy")
        if xcol and ycol and xcol != ycol:
            fig3, ax3 = plt.subplots(figsize=(3.5,2.5))
            sns.scatterplot(x=df[xcol], y=df[ycol], ax=ax3, alpha=0.6)
            ax3.set_title(f"{xcol} vs {ycol}", fontsize=10)
            ax3.tick_params(labelsize=8)
            st.pyplot(fig3, use_container_width=False)
            st.caption(f"Relationship between **{xcol}** and **{ycol}**.")

    if cat_cols:
        st.markdown("#### Categorical Distribution")
        chosen_cat = st.selectbox("Categorical column", options=cat_cols, key="catcol")
        ct = df[chosen_cat].value_counts().head(10)
        fig4, ax4 = plt.subplots(figsize=(3,2))
        sns.barplot(x=ct.values, y=ct.index, ax=ax4, palette="viridis")
        ax4.set_title(f"Top categories in {chosen_cat}", fontsize=10)
        ax4.tick_params(labelsize=8)
        st.pyplot(fig4, use_container_width=False)
        st.caption(generate_plot_description(df[[chosen_cat]], "bar"))

with tabs[2]:
    st.header("Correlations")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    corr_cols = st.multiselect("Choose numeric columns", options=num_cols, default=num_cols[:4])
    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr()
        c1, c2 = st.columns([2,1])
        with c1:
            fig, ax = plt.subplots(figsize=(3.5,2.5))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=ax, cbar=False)
            ax.set_title("Correlation Heatmap", fontsize=10)
            st.pyplot(fig, use_container_width=False)
        with c2:
            st.dataframe(corr.style.background_gradient(cmap="Blues"))
        st.caption(generate_plot_description(df[corr_cols], "heatmap"))
    else:
        st.info("Select at least two numeric columns.")

with tabs[3]:
    st.header("Time Series")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    datetime_cols = [c for c in df.columns if "date" in c.lower() or pd.api.types.is_datetime64_any_dtype(df[c])]
    if datetime_cols:
        dt = st.selectbox("Datetime column", options=datetime_cols, key="dt")
        ts_col = st.selectbox("Numeric column", options=[None]+num_cols, key="ts")
        if ts_col:
            temp = df.copy()
            temp[dt] = pd.to_datetime(temp[dt], errors="coerce")
            series = temp.groupby(pd.Grouper(key=dt, freq="D"))[ts_col].mean().dropna()
            fig, ax = plt.subplots(figsize=(3.5,2.5))
            ax.plot(series.index, series.values, linewidth=1)
            ax.set_title(f"{ts_col} over time", fontsize=10)
            ax.tick_params(labelsize=8)
            st.pyplot(fig, use_container_width=False)
            st.caption(f"Trend of **{ts_col}** aggregated daily by `{dt}`.")
    else:
        st.info("No datetime columns detected.")

with tabs[4]:
    st.header("Top Categories")
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        selected = st.multiselect("Choose categorical columns", options=cat_cols, default=cat_cols[:2])
        if selected:
            cols = st.columns(len(selected))
            for i, col in enumerate(selected):
                vc = df[col].value_counts().head(10)
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(3,2))
                    sns.barplot(x=vc.values, y=[str(x) for x in vc.index], ax=ax, palette="viridis")
                    ax.set_title(f"Top categories in {col}", fontsize=10)
                    ax.tick_params(labelsize=8)
                    st.pyplot(fig, use_container_width=False)
                    st.caption(generate_plot_description(df[[col]], "bar"))
    else:
        st.info("No categorical columns found.")

with tabs[5]:
    st.header("Auto EDA Report")
    if st.button("Generate EDA summary"):
        with st.spinner("Generating report..."):
            try:
                docx_bytes = analyze_and_report(df)
                st.success("Report ready.")
                st.download_button("Download EDA_Report.docx", data=docx_bytes,
                                   file_name="EDA_Report.docx",
                                   mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            except Exception as e:
                st.error(f"Failed: {e}")

with tabs[6]:
    st.header("Q&A Chatbot")
    user_q = st.text_input("Ask about your dataset:")
    if st.button("Ask"):
        if user_q:
            st.markdown("**Answer:**")
            st.write(fallback_qa(user_q, df))

st.caption("Built with ❤️ Streamlit + pandas + seaborn + python-docx by @Sumit_Kumar_PGA43")

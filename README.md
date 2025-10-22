# 📊 Professional AI-Powered EDA Dashboard

An interactive Streamlit-based dashboard for **Exploratory Data Analysis (EDA)** that:
- Uploads and explores CSV/Excel datasets.
- Provides **compact, grid-based visualizations** (distributions, correlations, time series, categories).
- Generates a **professional DOCX EDA Report** with tables, charts, and storytelling.
- Includes a simple **Q&A chatbot** for dataset queries.

---

## 🚀 Features
- **Data Upload**: CSV/Excel support (auto-detects delimiters).
- **Overview Tab**: Dataset preview, data types, missing values.
- **Distributions Tab**: Histograms, boxplots, scatterplots with AI storytelling.
- **Correlations Tab**: Heatmap + styled correlation table.
- **Time Series Tab**: Trends for numeric columns over time.
- **Top Categories Tab**: Bar charts for categorical distributions.
- **EDA Report Tab**: Export a detailed `.docx` report with:
  - Executive summary
  - Dataset overview
  - Numeric summary (tabular)
  - Categorical summary (tabular)
  - Correlation heatmap
  - Distributions & boxplots
  - Categorical charts
  - Insights + conclusion
- **Q&A Chatbot**: Basic dataset Q&A (rows, columns, missing values, top categories).

---

## 🛠️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-eda-dashboard.git
cd ai-eda-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, install manually:
```bash
pip install streamlit pandas numpy matplotlib seaborn python-docx
```

---

## ▶️ Usage

### Run the dashboard
```bash
streamlit run streamlit_app.py
```

### Workflow
1. Upload your dataset (`CSV` or `Excel`).
2. Explore insights via tabs (Overview → Distributions → Correlations → etc.).
3. Generate a **DOCX report** from the **EDA Report tab**.
4. Ask questions in the **Q&A Chatbot tab**.

---

## 📂 Project Structure
```
├── streamlit_app.py          # Main Streamlit dashboard
├── eda_report_notebook.py    # EDA report generator (DOCX)
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies (optional)
```

---

## 📑 Example Report
The generated report includes:
- Executive Summary
- Dataset Overview Table
- Numeric Summary (mean, std, quartiles, etc.)
- Categorical Summary (value counts)
- Correlation Heatmap
- Distributions & Boxplots
- Top Categories
- Conclusion

---

## ✨ Future Improvements
- Add model training & evaluation modules.
- Enhance chatbot with GPT-based LLM for deeper insights.
- Export reports in PDF format.

---

## 👨‍💻 Author
Developed by **[Your Name]**  
📧 Contact: your.email@example.com  

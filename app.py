import streamlit as st
from sentiment_analysis import run_analysis_from_ui

st.set_page_config(
    page_title="Gold Market Sentiment Dashboard",
    layout="wide"
)

st.title("🟡 Gold Market Sentiment Analyzer")
st.write("LLM + RAG powered market sentiment dashboard")

# User input
queries_text = st.text_area(
    "Enter search queries (one per line)",
    value="gold market\ngold price\ngold forecast"
)

num_articles = st.slider(
    "Number of articles per query",
    min_value=1,
    max_value=10,
    value=5
)

queries = [q.strip() for q in queries_text.split("\n") if q.strip()]

# Run analysis
if st.button("Run Sentiment Analysis"):
    with st.spinner("Analyzing news sentiment..."):
        results = run_analysis_from_ui(queries, num_articles)

    st.success("Analysis completed!")

    for res in results:
        st.subheader(res["title"])
        st.write(f"**Sentiment:** {res['sentiment']}")
        st.write(f"**Explanation:** {res['explanation']}")
        st.divider()

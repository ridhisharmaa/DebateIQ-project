"""
streamlit_app/app.py
DebateIQ — Interactive Argument Mining Dashboard

Features:
  • Paste any text → live claim/premise/stance detection
  • Interactive argument tree (PyVis HTML or matplotlib fallback)
  • Confidence threshold slider
  • Sentence-level result table with colour coding
  • 5 preloaded demo examples
"""

import streamlit as st
import requests
import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import List

API_URL = "http://localhost:8000"

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DebateIQ — Argument Mining",
    page_icon="⚖️",
    layout="wide",
)

# ── demo examples ─────────────────────────────────────────────────────────────
DEMOS = {
    "Reddit r/CMV — Climate Change": {
        "topic": "climate change",
        "text": (
            "Climate change is the most urgent issue of our time. "
            "The scientific consensus is clear: human activities are causing global temperatures to rise. "
            "Studies show that CO2 levels have reached record highs in the past century. "
            "However, some argue that economic costs of green policies outweigh the benefits. "
            "The Paris Agreement, signed by 196 countries, represents global commitment to action. "
            "Renewable energy is now cheaper than coal in most of the world."
        ),
    },
    "Wikipedia debate — Universal Healthcare": {
        "topic": "universal healthcare",
        "text": (
            "Universal healthcare should be a right for all citizens. "
            "Countries with universal systems report better health outcomes on average. "
            "The US spends more per capita on healthcare than any other nation yet has worse outcomes. "
            "Critics argue universal healthcare creates long wait times and reduces innovation. "
            "Evidence from Canada and the UK shows that wait times are manageable. "
            "A single-payer system would eliminate administrative overhead."
        ),
    },
    "Political speech — Immigration": {
        "topic": "immigration policy",
        "text": (
            "Strict immigration policies are necessary to protect national security. "
            "Immigrants contribute significantly to GDP and fill critical labour shortages. "
            "Data shows that immigrants commit crimes at lower rates than native-born citizens. "
            "Open borders would overwhelm public services and social welfare systems. "
            "The economic contribution of immigrants to social security is well documented. "
            "Border security must be balanced with humanitarian obligations."
        ),
    },
    "Product review — iPhone": {
        "topic": "iPhone quality",
        "text": (
            "The iPhone is the best smartphone on the market today. "
            "Apple's ecosystem integration is unmatched by any competitor. "
            "The camera system produces professional-quality photos. "
            "However, the price is significantly higher than comparable Android devices. "
            "Battery life has improved dramatically in recent models. "
            "Repairability scores remain low compared to competitors."
        ),
    },
    "News op-ed — Capital Punishment": {
        "topic": "capital punishment",
        "text": (
            "Capital punishment should be abolished in all democratic societies. "
            "Studies consistently show no deterrent effect on murder rates. "
            "The risk of executing innocent people is an irreversible injustice. "
            "Supporters argue it provides closure for victims' families. "
            "The cost of death penalty cases exceeds that of life imprisonment. "
            "Racial and socioeconomic disparities in sentencing undermine its legitimacy."
        ),
    },
}

STANCE_COLORS = {"PRO": "#28a745", "CON": "#dc3545", "NONE": "#6c757d"}
TYPE_EMOJI    = {"CLAIM": "🔵", "PREMISE": "🟠", "NONE": "⚪"}


# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://em-content.zobj.net/source/twitter/376/balance-scale_2696-fe0f.png", width=60)
    st.title("DebateIQ")
    st.caption("Argument Mining with SBERT + BiLSTM")
    st.divider()

    demo_choice = st.selectbox("📚 Load demo example", ["— custom input —"] + list(DEMOS.keys()))
    st.divider()

    threshold = st.slider("Confidence threshold", 0.3, 0.95, 0.50, 0.05,
                          help="Only show predictions above this confidence")
    st.divider()
    st.markdown("**Tasks**")
    st.markdown("✅ Claim Detection\n✅ Premise Detection\n✅ Stance Classification\n✅ Argument Tree")
    st.divider()
    st.caption("API: " + API_URL)


# ── main area ─────────────────────────────────────────────────────────────────
st.title("⚖️ DebateIQ — Argument Mining")
st.markdown("*Detect claims, premises and stances in any argumentative text.*")

col_input, col_info = st.columns([2, 1])

with col_input:
    if demo_choice != "— custom input —":
        demo = DEMOS[demo_choice]
        default_text  = demo["text"]
        default_topic = demo["topic"]
    else:
        default_text  = ""
        default_topic = ""

    text_input  = st.text_area("📝 Paste your text here", value=default_text, height=200)
    topic_input = st.text_input("🏷️ Topic (optional)", value=default_topic)
    analyze_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)

with col_info:
    st.info(
        "**How it works**\n\n"
        "1. Text is split into sentences\n"
        "2. SBERT encodes each sentence\n"
        "3. BiLSTM captures context\n"
        "4. Three heads predict:\n"
        "   - Is it a **claim**?\n"
        "   - Is it a **premise**?\n"
        "   - Is it **PRO/CON/NONE**?"
    )

# ── analysis ──────────────────────────────────────────────────────────────────
if analyze_btn and text_input.strip():
    with st.spinner("Analyzing argument structure…"):
        try:
            response = requests.post(
                f"{API_URL}/analyze",
                json={
                    "text": text_input,
                    "topic": topic_input or None,
                    "confidence_threshold": threshold,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Cannot connect to API. Make sure the FastAPI server is running:\n"
                     "```\nuvicorn api.main:app --reload\n```")
            st.stop()
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    # ── summary metrics ───────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Summary")
    s = data["summary"]
    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Sentences",  s["total_sentences"])
    m2.metric("Claims",     s["claims"])
    m3.metric("Premises",   s["premises"])
    sc = s["stance_counts"]
    m4.metric("🟢 PRO",  sc.get("PRO", 0))
    m5.metric("🔴 CON",  sc.get("CON", 0))
    m6.metric("⚫ NONE", sc.get("NONE", 0))
    st.caption(f"Processing time: {data['processing_ms']} ms")

    # ── sentence results table ────────────────────────────────────────────────
    st.divider()
    st.subheader("📋 Sentence-Level Results")

    rows = []
    for r in data["sentences"]:
        node_type = "CLAIM" if r["is_claim"] else ("PREMISE" if r["is_premise"] else "—")
        rows.append({
            "#":          r["index"] + 1,
            "Sentence":   r["sentence"],
            "Type":       f"{TYPE_EMOJI.get(node_type, '⚪')} {node_type}",
            "Stance":     r["stance"],
            "Claim Conf": f"{r['claim_confidence']:.2f}",
            "Prem. Conf": f"{r['premise_confidence']:.2f}",
            "Stance Conf":f"{r['stance_confidence']:.2f}",
        })

    df = pd.DataFrame(rows)

    def color_stance(val):
        c = STANCE_COLORS.get(val, "#6c757d")
        return f"background-color: {c}22; color: {c}; font-weight: bold;"

    st.dataframe(
        df.style.applymap(color_stance, subset=["Stance"]),
        use_container_width=True,
        height=300,
    )

    # ── stance distribution donut ─────────────────────────────────────────────
    col_donut, col_bar = st.columns(2)

    with col_donut:
        st.subheader("Stance Distribution")
        sc_data = s["stance_counts"]
        fig_donut = go.Figure(go.Pie(
            labels=list(sc_data.keys()),
            values=list(sc_data.values()),
            hole=0.5,
            marker_colors=[STANCE_COLORS.get(k, "#999") for k in sc_data],
        ))
        fig_donut.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_bar:
        st.subheader("Confidence per Sentence")
        conf_df = pd.DataFrame([{
            "idx":    r["index"] + 1,
            "Stance": r["stance"],
            "conf":   r["stance_confidence"],
        } for r in data["sentences"]])
        fig_bar = px.bar(conf_df, x="idx", y="conf", color="Stance",
                         color_discrete_map=STANCE_COLORS,
                         labels={"idx": "Sentence #", "conf": "Confidence"},
                         height=300)
        fig_bar.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── argument tree ─────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🌳 Argument Tree")

    tree = data["argument_tree"]
    nodes_data = tree.get("nodes", [])
    edges_data = tree.get("edges", [])

    if nodes_data:
        # build plotly network graph
        import networkx as nx
        G = nx.DiGraph()
        for n in nodes_data:
            G.add_node(n["id"], **n)
        for e in edges_data:
            G.add_edge(e["source"], e["target"], **e)

        pos = nx.spring_layout(G, seed=42, k=3)

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        type_color_map = {"CLAIM": "#4C72B0", "PREMISE": "#DD8452", "NONE": "#AAAAAA"}
        for nid in G.nodes():
            nd = G.nodes[nid]
            x, y = pos[nid]
            node_x.append(x); node_y.append(y)
            node_text.append(f"[{nd.get('node_type','?')}] {nd.get('text','')[:60]}")
            node_color.append(type_color_map.get(nd.get("node_type", "NONE"), "#AAA"))
            node_size.append(20 if nd.get("node_type") == "CLAIM" else 12)

        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]; x1, y1 = pos[v]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        fig_tree = go.Figure()
        fig_tree.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                      line=dict(width=1, color="#888"), hoverinfo="none"))
        fig_tree.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                      text=node_text,
                                      textposition="top center",
                                      marker=dict(size=node_size, color=node_color,
                                                  line=dict(width=1, color="white")),
                                      hoverinfo="text"))
        fig_tree.update_layout(showlegend=False, height=450,
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                margin=dict(t=10, b=10))
        st.plotly_chart(fig_tree, use_container_width=True)

        legend_col1, legend_col2, legend_col3 = st.columns(3)
        legend_col1.markdown("🔵 **Claim node**")
        legend_col2.markdown("🟠 **Premise node**")
        legend_col3.markdown("⚪ **Neutral node**")

    # ── raw JSON expander ─────────────────────────────────────────────────────
    with st.expander("📦 Raw API Response (JSON)"):
        st.json(data)

elif analyze_btn:
    st.warning("Please enter some text to analyze.")

else:
    st.markdown("""
    ---
    ### 👋 Getting Started
    1. Select a **demo example** from the sidebar or paste your own text
    2. Optionally enter the **topic** being debated
    3. Adjust the **confidence threshold** as needed
    4. Click **Analyze** to see claims, premises, stances and the argument tree
    """)

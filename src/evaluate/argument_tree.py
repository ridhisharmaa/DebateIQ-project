"""
src/evaluate/argument_tree.py
Build and visualise argument trees from model predictions.
Nodes = sentences  |  edges = support/attack relations
Root = main claim  |  Children = premises with stance colour
"""

import json
import numpy as np
import networkx as nx
from pathlib import Path
from typing import List, Dict, Optional

RESULTS = Path("results")
RESULTS.mkdir(parents=True, exist_ok=True)

STANCE_COLOR = {
    "PRO":  "#55A868",   # green
    "CON":  "#C44E52",   # red
    "NONE": "#8C8C8C",   # grey
}


# ── build graph ───────────────────────────────────────────────────────────────
def build_argument_graph(sentences: List[str],
                          claim_preds: np.ndarray,
                          premise_preds: np.ndarray,
                          stance_preds: np.ndarray,
                          stance_classes: List[str],
                          confidences: Optional[np.ndarray] = None) -> nx.DiGraph:
    """
    Constructs a directed argument graph:
      - Root node = first predicted claim
      - Premises point to their nearest claim (simplified heuristic)
    """
    G = nx.DiGraph()
    stance_labels = [stance_classes[s] for s in stance_preds]

    claim_indices = [i for i, c in enumerate(claim_preds) if c == 1]
    if not claim_indices:
        claim_indices = [0]   # fallback

    # add all nodes
    for i, sent in enumerate(sentences):
        label = "CLAIM" if claim_preds[i] else ("PREMISE" if premise_preds[i] else "NONE")
        conf  = float(confidences[i]) if confidences is not None else 1.0
        G.add_node(i,
                   text=sent[:80] + "…" if len(sent) > 80 else sent,
                   node_type=label,
                   stance=stance_labels[i],
                   confidence=round(conf, 3))

    # connect premises to nearest claim
    for i in range(len(sentences)):
        if premise_preds[i] == 1:
            nearest_claim = min(claim_indices, key=lambda c: abs(c - i))
            G.add_edge(i, nearest_claim,
                       stance=stance_labels[i],
                       color=STANCE_COLOR.get(stance_labels[i], "#999"))

    return G


# ── matplotlib visualisation ──────────────────────────────────────────────────
def plot_argument_tree_mpl(G: nx.DiGraph, title: str = "Argument Tree", save: bool = True):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    pos = nx.spring_layout(G, seed=42, k=2.5)

    claim_nodes   = [n for n, d in G.nodes(data=True) if d["node_type"] == "CLAIM"]
    premise_nodes = [n for n, d in G.nodes(data=True) if d["node_type"] == "PREMISE"]
    other_nodes   = [n for n, d in G.nodes(data=True) if d["node_type"] == "NONE"]

    edge_colors = [d.get("color", "#999") for _, _, d in G.edges(data=True)]

    fig, ax = plt.subplots(figsize=(14, 9))
    nx.draw_networkx_nodes(G, pos, nodelist=claim_nodes,   node_color="#4C72B0",
                           node_size=1200, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=premise_nodes, node_color="#DD8452",
                           node_size=800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=other_nodes,   node_color="#AAAAAA",
                           node_size=400, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, arrows=True,
                           arrowsize=20, width=2, ax=ax)

    labels = {n: f"[{d['node_type'][0]}]\n{d['text'][:40]}…"
              if len(d['text']) > 40 else f"[{d['node_type'][0]}]\n{d['text']}"
              for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=7, ax=ax)

    patches = [
        mpatches.Patch(color="#4C72B0", label="Claim"),
        mpatches.Patch(color="#DD8452", label="Premise"),
        mpatches.Patch(color="#55A868", label="PRO edge"),
        mpatches.Patch(color="#C44E52", label="CON edge"),
    ]
    ax.legend(handles=patches, loc="upper right")
    ax.set_title(title, fontsize=14)
    ax.axis("off")
    plt.tight_layout()

    if save:
        plt.savefig(RESULTS / "argument_tree.png", dpi=150)
        print("[tree] Saved → results/argument_tree.png")
    plt.show()
    return fig


# ── interactive PyVis HTML ────────────────────────────────────────────────────
def export_interactive_html(G: nx.DiGraph,
                             output_path: str = "results/argument_tree.html"):
    try:
        from pyvis.network import Network
    except ImportError:
        print("[tree] pyvis not installed — skipping interactive export.")
        return

    net = Network(height="750px", width="100%", directed=True,
                  bgcolor="#1a1a2e", font_color="white")
    net.barnes_hut()

    type_color = {"CLAIM": "#4C72B0", "PREMISE": "#DD8452", "NONE": "#888888"}

    for node_id, data in G.nodes(data=True):
        color = type_color.get(data["node_type"], "#888")
        title = (f"<b>{data['node_type']}</b><br>"
                 f"Stance: {data['stance']}<br>"
                 f"Conf: {data['confidence']}<br>"
                 f"{data['text']}")
        net.add_node(node_id, label=data['text'][:30], color=color,
                     title=title, size=20 if data["node_type"] == "CLAIM" else 12)

    for src, dst, edata in G.edges(data=True):
        net.add_edge(src, dst, color=edata.get("color", "#999"),
                     title=edata.get("stance", ""))

    net.save_graph(output_path)
    print(f"[tree] Interactive HTML → {output_path}")


# ── JSON export for API ───────────────────────────────────────────────────────
def graph_to_json(G: nx.DiGraph) -> dict:
    return {
        "nodes": [
            {"id": n, **{k: v for k, v in d.items()}}
            for n, d in G.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, **{k: val for k, val in d.items()}}
            for u, v, d in G.edges(data=True)
        ],
    }

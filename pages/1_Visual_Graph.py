import streamlit as st
import sqlite3
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(layout="wide", page_title="Visual Graph Explorer")
st.title("🕸️ Full-Screen Visual Graph Explorer")
st.markdown("Search for an entity below to visualize its network.")

@st.cache_resource
def load_db():
    return sqlite3.connect('data/outputs/rag_kg.db', check_same_thread=False)

conn = load_db()
c = conn.cursor()

entity = st.text_input("Enter an entity to visualize (e.g., 'CRAG', 'BERT'):")
if entity:
    # Use LIKE for case-insensitive, partial matching since entities are stored in mixed case
    search_pattern = f"%{entity}%"
    c.execute(
        "SELECT subject, type, object, evidence, paper FROM relations "
        "WHERE subject LIKE ? OR object LIKE ? LIMIT 40",
        (search_pattern, search_pattern)
    )
    rels = c.fetchall()
    
    if len(rels) == 40:
        st.warning("⚠️ Showing the first 40 connections to prevent the graph from becoming a chaotic hairball.")
        
    if rels:
        nodes = []
        edges = []
        added_nodes = set()
        
        for rel in rels:
            source, relation, target, evidence, paper = rel[0], rel[1], rel[2], rel[3], rel[4]
            
            if source not in added_nodes:
                nodes.append(Node(id=source, label=source, size=25, color="#ff4b4b"))
                added_nodes.add(source)
                
            if target not in added_nodes:
                nodes.append(Node(id=target, label=target, size=25, color="#00cc96"))
                added_nodes.add(target)
                
            edges.append(Edge(source=source, target=target, title=f"Type: {relation} | Source: {paper} | Evidence: {evidence}"))
            
        config = Config(width=1200, height=800, directed=True, physics=True, hierarchical=False)
        agraph(nodes=nodes, edges=edges, config=config)
    else:
        st.warning("No relations found for this entity.")

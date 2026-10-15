import streamlit as st
import sqlite3
import pickle

st.set_page_config(layout="wide", page_title="RAG Knowledge Graph")

@st.cache_resource
def load_db():
    conn = sqlite3.connect('data/outputs/rag_kg.db', check_same_thread=False)
    with open('data/outputs/rag_kg.gpickle', 'rb') as f:
        kg = pickle.load(f)
    return conn, kg

def load_llm(model_name="llama3"):
    import requests
    def generate_answer(prompt, **kwargs):
        try:
            res = requests.post("http://localhost:11434/api/generate", json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }).json()
            if 'response' in res:
                return [{'generated_text': res['response']}]
            else:
                return [{'generated_text': f"Ollama Error: {res}"}]
        except Exception as e:
            return [{'generated_text': f"Error connecting to Ollama: {str(e)}. Make sure Ollama is running!"}]
            
    return generate_answer

st.title("📚 Scientific Knowledge Graph for RAG Papers")
st.markdown("This interface explores the Knowledge Graph extracted from core RAG research papers.")

try:
    conn, kg = load_db()
    c = conn.cursor()
except Exception as e:
    st.warning("Database not found. Please run scripts 01 through 03 first.")
    st.stop()

# Basic Stats
c.execute("SELECT COUNT(*) FROM entities")
num_ents = c.fetchone()[0]
c.execute("SELECT COUNT(*) FROM relations")
num_rels = c.fetchone()[0]

st.markdown(f"**Knowledge Base Stats:** `{num_ents}` Entities | `{num_rels}` Relations")

st.info("👈 **Navigate to the '1 Visual Graph' page in the left sidebar to open the Full-Screen Interactive Graph!**")

st.divider()

# RAG QA Interface
st.subheader("🤖 Ask the GraphRAG (LLM Integration)")
st.info("The local LLM will traverse the graph, pull citations, and generate an answer based purely on the evidence.")
question = st.text_input("Ask a plain-English question (e.g., 'What does CRAG evaluate on?'):")

if question:
    with st.spinner("Connecting to local Ollama instance..."):
        # You can change 'llama3' to whichever model you have downloaded in Ollama!
        llm = load_llm(model_name="llama3")
        
    import re
    # Find entities in question to anchor the graph search
    question_lower = question.lower()
    c.execute("SELECT name FROM entities")
    all_ents = [row[0] for row in c.fetchall()]
    
    stopwords = {"what", "when", "where", "which", "who", "whom", "how", "why", "this", "that", "these", "those", "have", "has", "had", "does", "do", "did", "are", "is"}
    
    found_ents = set()
    for ent in all_ents:
        if len(ent) >= 3 and ent.lower() not in stopwords:
            # Use whole word matching to prevent "rag" matching inside "fragment"
            pattern = r'\b' + re.escape(ent.lower()) + r'\b'
            if re.search(pattern, question_lower):
                found_ents.add(ent)
            
    evidence_list = []
    if not found_ents:
        st.warning("⚠️ Could not find exact mathematical graph entities for your question. Falling back to keyword evidence search...")
        # Fallback keyword search on the evidence directly
        words = [w for w in question_lower.replace('?', '').split() if w not in stopwords and len(w) >= 3]
        for w in words:
            c.execute("SELECT evidence, paper FROM relations WHERE evidence LIKE ? LIMIT 10", (f"%{w}%",))
            evidence_list.extend([f"[Source: [{r[1]}](https://arxiv.org/abs/{r[1]})] {r[0]}" for r in c.fetchall()])
    else:
        st.write(f"**Graph Anchors Found:** `{', '.join(found_ents)}`")
        for ent in found_ents:
            c.execute("SELECT evidence, paper FROM relations WHERE subject=? OR object=? LIMIT 10", (ent, ent))
            evidence_list.extend([f"[Source: [{r[1]}](https://arxiv.org/abs/{r[1]})] {r[0]}" for r in c.fetchall()])
            
        # Deduplicate evidence string so the LLM doesn't read the same sentence 10 times
        evidence_list = list(set(evidence_list))[:15]
        context = "\n".join(evidence_list)
        
        if not context.strip():
            st.warning("No relational evidence found in the graph for these anchors.")
        else:
            prompt = f"Based on this research context:\n{context}\n\nAnswer the question with a detailed, complete sentence: {question}"
            
            with st.spinner("LLM is reasoning over the graph evidence..."):
                ans = llm(prompt, max_length=256, truncation=True)[0]['generated_text']
            
            st.success("### LLM Answer")
            st.write(ans)
            
            with st.expander("View Graph Citations Used by LLM"):
                for ev in evidence_list:
                    st.info(ev)

st.divider()
st.subheader("🔍 Manual Entity Explorer")
entity_query = st.text_input("Enter an entity to view its raw detailed relations:")
if entity_query:
    c.execute("SELECT name, type, papers FROM entities WHERE name LIKE ? LIMIT 5", ('%'+entity_query+'%',))
    results = c.fetchall()
    
    if results:
        for r in results:
            st.markdown(f"### {r[0]} ({r[1]})")
            
            # Show edges
            c.execute("SELECT subject, type, object, evidence, paper FROM relations WHERE subject=? OR object=? LIMIT 50", (r[0], r[0]))
            rels = c.fetchall()
            
            if rels:
                unique_rels = {}
                for rel in rels:
                    k = f"{rel[0]}|{rel[1]}|{rel[2]}"
                    # If this relation exists, only replace if the evidence is longer/more detailed
                    if k not in unique_rels or len(rel[3]) > len(unique_rels[k][3]):
                        unique_rels[k] = rel
                
                st.write("#### Detailed Evidence")
                for k, rel in unique_rels.items():
                    with st.container(border=True):
                        st.markdown(f"🔵 **{rel[0]}** ➡️ [{rel[1]}] ➡️ 🟢 **{rel[2]}**")
                        st.info(f'"{rel[3]}"\n\n*(Source Paper: [{rel[4]}](https://arxiv.org/abs/{rel[4]}))*')
    else:
        st.info("No entities found.")

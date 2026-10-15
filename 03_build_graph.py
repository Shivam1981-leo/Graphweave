import json
import networkx as nx
import sqlite3
import pickle
from sentence_transformers import SentenceTransformer

def build_graph():
    print("Loading extracted data...")
    with open("data/outputs/entities.json", "r") as f:
        entities = json.load(f)
    with open("data/outputs/relations.json", "r") as f:
        relations = json.load(f)
        
    # 1. Build NetworkX Graph
    kg = nx.MultiDiGraph()
    for name, data in entities.items():
        kg.add_node(name, type=data['type'], mentions=data['mentions'])
        
    for rel in relations:
        kg.add_edge(rel['subject'], rel['object'], 
                    type=rel['type'], evidence=rel['evidence'], paper=rel['paper'])
                    
    with open('data/outputs/rag_kg.gpickle', 'wb') as f:
        pickle.dump(kg, f, pickle.HIGHEST_PROTOCOL)
    print(f"Graph built with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges.")
    
    # 2. Create Embeddings
    print("Generating embeddings using SentenceTransformers...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    entity_texts = list(entities.keys())
    embeddings = model.encode(entity_texts)
    
    # 3. Store in SQLite
    print("Saving to SQLite database...")
    conn = sqlite3.connect('data/outputs/rag_kg.db')
    c = conn.cursor()
    
    c.execute('DROP TABLE IF EXISTS entities')
    c.execute('DROP TABLE IF EXISTS relations')
    
    c.execute('''CREATE TABLE entities
                 (name TEXT PRIMARY KEY, type TEXT, papers TEXT, embedding BLOB)''')
    
    c.execute('''CREATE TABLE relations
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, subject TEXT, object TEXT, 
                  type TEXT, evidence TEXT, paper TEXT)''')
                  
    for i, name in enumerate(entity_texts):
        emb_bytes = embeddings[i].tobytes()
        papers_str = ",".join(entities[name]['papers'])
        c.execute("INSERT INTO entities VALUES (?, ?, ?, ?)", 
                  (name, entities[name]['type'], papers_str, emb_bytes))
                  
    for rel in relations:
        c.execute("INSERT INTO relations (subject, object, type, evidence, paper) VALUES (?, ?, ?, ?, ?)",
                  (rel['subject'], rel['object'], rel['type'], rel['evidence'], rel['paper']))
                  
    conn.commit()
    conn.close()
    print("Database created successfully.")

if __name__ == "__main__":
    build_graph()

import json
import requests
import os
import re

os.makedirs("data/outputs", exist_ok=True)

# Point to local Ollama server
OLLAMA_API = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3"

def extract_with_ollama(text):
    prompt = f"""
You are a precise data extraction AI. Extract the knowledge graph from the following scientific text.
Focus on identifying Methods, Datasets, and Metrics.
Extract how they relate to each other (e.g. USES, EVALUATES_ON, IMPROVES).

Respond ONLY with valid JSON in this exact structure:
{{
  "entities": [
    {{"name": "CRAG", "type": "METHOD"}},
    {{"name": "FEVER", "type": "DATASET"}}
  ],
  "relations": [
    {{"subject": "CRAG", "relation": "EVALUATES_ON", "object": "FEVER", "evidence": "CRAG was evaluated on the FEVER dataset."}}
  ]
}}
Do NOT wrap the JSON in markdown blocks like ```json.
Text to extract from:
{text}
"""
    try:
        res = requests.post(OLLAMA_API, json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "format": "json", # Force Ollama to strictly output valid JSON
            "options": {"temperature": 0.0} # enforce deterministic JSON output
        }).json()
        
        # Clean markdown if ollama wraps it anyway
        raw = res.get('response', '{}')
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'^```\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        
        return json.loads(raw.strip())
    except Exception as e:
        print(f"Ollama Extraction Error or JSON parsing failed: {e}")
        return {"entities": [], "relations": []}

def extract_knowledge(corpus_path):
    with open(corpus_path, "r", encoding="utf-8") as f:
        papers = json.load(f)
        
    all_entities = {}
    all_relations = []
    
    # We will process 20 papers, taking the first 8000 characters of each to stay within context windows
    for i, paper in enumerate(papers):
        print(f"[{i+1}/{len(papers)}] Using Ollama {MODEL_NAME} to process paper: {paper['title']}")
        text = paper['text'][:8000] 
        
        kg = extract_with_ollama(text)
        
        # Merge entities
        for ent in kg.get("entities", []):
            word = ent.get("name")
            if not word: continue
            if word not in all_entities:
                all_entities[word] = {"text": word, "type": ent.get("type", "CONCEPT"), "papers": set(), "mentions": 0}
            all_entities[word]["papers"].add(paper['id'])
            all_entities[word]["mentions"] += 1
            
        # Merge relations
        for rel in kg.get("relations", []):
            if rel.get("subject") and rel.get("object"):
                all_relations.append({
                    "subject": rel["subject"],
                    "object": rel["object"],
                    "type": rel.get("relation", "RELATES_TO"),
                    "evidence": rel.get("evidence", "Ollama inferred this relationship from the text."),
                    "paper": paper['id']
                })

    # Convert sets to lists
    for k, v in all_entities.items():
        v["papers"] = list(v["papers"])
        
    with open("data/outputs/entities.json", "w") as f:
        json.dump(all_entities, f, indent=2)
    with open("data/outputs/relations.json", "w") as f:
        json.dump(all_relations, f, indent=2)
        
    print(f"Extraction Complete! Extracted {len(all_entities)} entities and {len(all_relations)} relations using Ollama.")

if __name__ == "__main__":
    print("WARNING: Make sure your local Ollama server is running (ollama serve)!")
    extract_knowledge("data/processed/papers_corpus.json")

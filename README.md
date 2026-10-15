# 🕸️ GraphWeave: Scientific Knowledge Graph & GraphRAG for Research Papers

GraphWeave is a scientific Knowledge Graph (KG) and Graph-augmented Retrieval (GraphRAG) system designed to extract, interlink, explore, and reason over concepts, methods, datasets, and metrics across research papers.

By combining local Large Language Models (LLMs via Ollama), directed knowledge graphs (NetworkX), semantic vector embeddings (SentenceTransformers), and relational persistence (SQLite), GraphWeave enables verifiable, citation-grounded question answering and interactive network visualization.

---

## 🚀 Key Features

- **Automated Knowledge Extraction (`02_extract_kg.py`)**: Uses local LLMs (Llama 3 via Ollama) to extract structured entities (`METHOD`, `DATASET`, `METRIC`, `CONCEPT`) and directional relationships (`EVALUATES_ON`, `USES`, `IMPROVES`, etc.) directly from research texts with strict JSON formatting.
- **Graph Construction & Embedding Generation (`03_build_graph.py`)**:
  - Compiles extracted entities and relationships into a directed multi-graph (`NetworkX MultiDiGraph`).
  - Computes 384-dimensional dense semantic embeddings for entities using `SentenceTransformers` (`all-MiniLM-L6-v2`).
  - Persists relational data and vector representations into an indexed SQLite database (`rag_kg.db`) and serialized graph pickle (`rag_kg.gpickle`).
- **Interactive Streamlit Web Dashboard (`04_app.py`)**:
  - **GraphRAG Question Answering**: Plain-English queries are grounded against graph evidence; the LLM traverses connected nodes, gathers verified citations, and synthesizes grounded answers with direct links to arXiv source papers.
  - **Entity Explorer**: Query and inspect entity relations, connection degrees, and context snippets.
- **Full-Screen Network Visualizer (`pages/1_Visual_Graph.py`)**:
  - Interactive force-directed network graph powered by `streamlit-agraph`, allowing real-time entity search, node exploration, and edge inspection.

---

## 📂 Repository Structure

```plaintext
Graphweave/
│
├── 02_extract_kg.py          # LLM-based entity & relation extraction pipeline (Ollama)
├── 03_build_graph.py         # NetworkX graph compilation, embedding generation & SQLite export
├── 04_app.py                 # Main Streamlit web application & GraphRAG QA interface
│
├── pages/
│   └── 1_Visual_Graph.py     # Full-screen interactive graph visualizer (Streamlit-Agraph)
│
├── data/
│   ├── raw_pdfs/             # Source research papers in PDF format
│   ├── processed/            # Extracted paper corpus (papers_corpus.json)
│   └── outputs/              # Knowledge graph outputs:
│       ├── entities.json     # Extracted entity dictionary
│       ├── relations.json    # Extracted relational triples with evidence
│       ├── rag_kg.db         # SQLite database storing entities, relations & embeddings
│       └── rag_kg.gpickle    # Serialized NetworkX MultiDiGraph
│
├── unpacked_paper/           # Unpacked paper structure and XML components
├── app_run.txt               # Quick reference commands for launching the application
└── requirements.txt          # Python dependencies
```

---

## 🛠️ Technology Stack

| Component | Technology | Description |
|---|---|---|
| **Local LLM Inference** | [Ollama](https://ollama.ai/) (`llama3`) | Privacy-first local inference for extraction & reasoning |
| **Graph Structure** | [NetworkX](https://networkx.org/) | Directed multi-graph representation and node/edge attributes |
| **Dense Embeddings** | [Sentence-Transformers](https://sbert.net/) (`all-MiniLM-L6-v2`) | Entity semantic representation and similarity mapping |
| **Relational Storage** | [SQLite](https://sqlite.org/) | Lightweight database storing structured graph triples & embeddings |
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Web dashboard for GraphRAG Q&A and exploration |
| **Graph Visualization** | [streamlit-agraph](https://github.com/ChrisChross/streamlit-agraph) | Interactive force-directed physics graph in the browser |

---

## ⚙️ Installation & Setup

### 1. Prerequisites
Ensure you have Python 3.10+ installed and Ollama running locally.

- Download and install [Ollama](https://ollama.ai/).
- Pull the Llama 3 model:
  ```bash
  ollama pull llama3
  ollama serve
  ```

### 2. Clone the Repository
```bash
git clone https://github.com/Shivam1981-leo/Graphweave.git
cd Graphweave
```

### 3. Create & Activate a Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚦 Execution Workflow

### Step 1: Knowledge Graph Extraction
Run the extraction script to read processed research papers and extract entities/relationships using Ollama:
```bash
python 02_extract_kg.py
```
*Outputs: `data/outputs/entities.json` and `data/outputs/relations.json`.*

### Step 2: Build the Graph & Database
Compile the extracted components into the NetworkX graph and generate SQLite tables with vector embeddings:
```bash
python 03_build_graph.py
```
*Outputs: `data/outputs/rag_kg.db` and `data/outputs/rag_kg.gpickle`.*

### Step 3: Launch the Streamlit App
Start the interactive application:
```bash
streamlit run 04_app.py
```

Open `http://localhost:8501` in your browser. From the sidebar, you can navigate between:
1. **RAG Knowledge Graph**: Natural language query engine and manual entity search with paper citations.
2. **1 Visual Graph**: Dynamic, interactive visual node-link diagram.

---

## 📄 License
This project is licensed under the MIT License.

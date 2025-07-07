from langchain_community.document_loaders import TextLoader, PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma  # NEW import location
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

import spacy
import networkx as nx
import matplotlib.pyplot as plt
import os
import re
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
import whisper  # This will now import the correct OpenAI Whisper


print("[i] Loading MOSDAC documents...")

try:
    text_loader = DirectoryLoader(
        path='./mosdac_data/processed',
        glob='**/*.txt',
        loader_cls=lambda path: TextLoader(path, encoding='utf-8'),
        show_progress=True
    )
    text_docs = text_loader.load()
    if not text_docs:
        print("[!] No text documents loaded from ./mosdac_data/processed")
except Exception as e:
    raise RuntimeError(f"[FATAL] Failed to load text documents: {e}")

try:
    pdf_loader = DirectoryLoader(
        path='./mosdac_data/processed',
        glob='**/*.pdf',
        loader_cls=PyMuPDFLoader,
        show_progress=True
    )
    pdf_docs = pdf_loader.load()
    if not pdf_docs:
        print("[!] No PDF documents loaded from ./mosdac_data/processed")
except Exception as e:
    raise RuntimeError(f"[FATAL] Failed to load PDF documents: {e}")

all_docs = text_docs + pdf_docs
if not all_docs:
    raise RuntimeError("[FATAL] No documents found in ./mosdac_data/processed. Please add .txt or .pdf files.")

print(f"[✓] Loaded {len(all_docs)} documents.")

try:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(all_docs)
    if not split_docs:
        raise RuntimeError("[FATAL] No document chunks created. Check your input files.")
except Exception as e:
    raise RuntimeError(f"[FATAL] Failed to split documents: {e}")

print(f"[✓] Split into {len(split_docs)} chunks.")

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
except Exception as e:
    raise RuntimeError(f"[FATAL] Failed to load HuggingFaceEmbeddings: {e}")

try:
    vector_store = Chroma.from_documents(split_docs, embeddings)
except Exception as e:
    raise RuntimeError(f"[FATAL] Failed to create Chroma vector store: {e}")

try:
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})
    if retriever is None:
        raise RuntimeError("[FATAL] Retriever could not be initialized.")
except Exception as e:
    raise RuntimeError(f"[FATAL] Failed to create retriever: {e}")

try:
    llm = ChatGoogleGenerativeAI(
        model='gemini-1.5-flash',
        temperature=0.2,
        api_key='AIzaSyBlVAGaEchXF46DbcNFE66X6HbbcN4oSXI'
    )
    if llm is None:
        raise RuntimeError("[FATAL] LLM could not be initialized.")
except Exception as e:
    raise RuntimeError(f"[FATAL] Failed to initialize LLM: {e}")

try:
    prompt = PromptTemplate(
        input_variables=["context", "query"],
        template=(
            "You are a knowledgeable assistant. Use both the provided MOSDAC/ISRO context and your own knowledge "
            "to answer thoroughly. If the question is about a satellite, mention: Name, Function, Launch Date, and Applications. "
            "If it is about a cyclone or event, mention: Event Name, Date(s), Regions affected, and Organization response.\n\n"
            "Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
        )
    )
    if prompt is None:
        raise RuntimeError("[FATAL] Prompt could not be initialized.")
except Exception as e:
    raise RuntimeError(f"[FATAL] Failed to initialize prompt: {e}")

os.environ["PATH"] += os.pathsep + r"C:/Users/arnav/Downloads/ffmpeg-7.1.1-essentials_build/ffmpeg-7.1.1-essentials_build/bin"


nlp = spacy.load("en_core_web_sm")
whisper_model = whisper.load_model("base")



def extract_entities(answer_text):
    entities = set()
    doc = nlp(answer_text)
    for ent in doc.ents:
        entities.add((ent.text.strip(), ent.label_))

    key_val_pattern = re.compile(r"(?P<key>[\w\s\/\-\(\)]+)\s*[:=]\s*(?P<value>[^\n]+)")
    for match in key_val_pattern.finditer(answer_text):
        key = match.group("key").strip().title()
        value = match.group("value").strip()
        if value and len(value.split()) < 12:
            label = key.upper().replace(" ", "_")
            entities.add((value, label))

    try:
        chunks = ne_chunk(pos_tag(word_tokenize(answer_text)))
        for chunk in chunks:
            if isinstance(chunk, Tree):
                phrase = " ".join(c[0] for c in chunk.leaves())
                entities.add((phrase, "NP_PHRASE"))
    except Exception as e:
        print(f"[!] NLTK fallback failed: {e}")

    return list(entities)

def build_and_save_kg(entities, filename="knowledge_graph.png"):
    if not entities:
        print("[!] No entities found.")
        return

    G = nx.Graph()

    preferred_roots = ["STATION", "SATELLITE", "DISASTER", "SENSOR"]
    root_entity = next((ent for ent in entities if ent[1] in preferred_roots), entities[0])
    main_entity = root_entity[0]

    for ent_text, ent_label in entities:
        G.add_node(ent_text, label=ent_label)
        if ent_text != main_entity:
            G.add_edge(main_entity, ent_text, relation=ent_label)

    plt.figure(figsize=(15, 10))
    plt.style.use('dark_background')
    pos = nx.spring_layout(G, k=1.5, iterations=50)

    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=2500, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color='white', width=2, alpha=0.6)

    labels = {node: f"{node}\n({data['label']})" for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, font_color='black')

    edge_labels = {(u, v): d.get("relation", "") for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, font_color='yellow')

    plt.title("Knowledge Graph from RAG Answer", color='white', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#222222')
    plt.close()

    print(f"[✓] Knowledge graph saved to {filename}")

def process_query(query):
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.format(context=context, query=query)
    response = llm.invoke(final_prompt)
    answer = response.content.strip()
    entities = extract_entities(answer)
    return context, answer, entities

# Do not run any input loop or main code here!
    docs = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in docs])
    final_prompt = prompt.format(context=context, query=query)
    response = llm.invoke(final_prompt)
    answer = response.content.strip()
    entities = extract_entities(answer)
    return context, answer, entities

# Do not run any input loop or main code here!

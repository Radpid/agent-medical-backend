# ==============================================================================
# DATEI: main.py
# ROLLE: Implementiert eine zweigleisige Strategie:
#        - 'rapid' Modus: Ein schneller, dynamischer Plan.
#        - 'deep' Modus: Ein fortschrittlicher "Exploratory Graph"-Agent, der
#          ein Wissensnetz aufbaut und daraus eine tiefgehende Synthese erstellt.
# SPRACHE: Deutsch
# VERSION: 7.0.0
# ==============================================================================

import os
import sys
import logging
import asyncio
import json
import re
from typing import List, Optional, Literal, Dict, Any, AsyncGenerator

import httpx
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

# ==============================================================================
# TEIL 1: KONFIGURATION UND INITIALISIERUNG
# ==============================================================================
def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        root_logger.addHandler(stream_handler)

load_dotenv()
class Settings(BaseSettings):
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY")
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY")
    class Config: env_file = ".env"

settings = Settings()
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medizinischer Agent mit Exploratory Graph Logik",
    description="Ein KI-Agent, der je nach Modus unterschiedliche, hochentwickelte Recherchestrategien anwendet.",
    version="7.0.0"
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

try:
    json_generation_config = genai.GenerationConfig(response_mime_type="application/json")
    llm_json_model = genai.GenerativeModel('gemini-1.5-flash', generation_config=json_generation_config)
    llm_text_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Google Gemini-Modelle initialisiert.")
except Exception as e:
    logger.error(f"Fehler bei der Konfiguration der Gemini-API: {e}")
    llm_json_model, llm_text_model = None, None

# ==============================================================================
# TEIL 2: DATENMODELLE UND WERKZEUGE
# ==============================================================================
class ResearchRequest(BaseModel):
    query: str
    mode: Literal["rapid", "deep"]

async def execute_search_tool(query: str, site_filter: str = "") -> List[Dict[str, Any]]:
    # ... (Keine Änderungen hier)
    logger.info(f"Such-Werkzeug wird für '{query}' mit Filter '{site_filter}' verwendet")
    search_query = f"{query} {site_filter}"
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": search_query, "num": 5})
    headers = {'X-API-KEY': settings.SERPER_API_KEY, 'Content-Type': 'application/json'}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json().get('organic', [])
        except Exception as e:
            logger.error(f"Fehler im Such-Werkzeug: {e}")
            return []

async def scrape_tool(url: str) -> str:
    # ... (Keine Änderungen hier)
    logger.info(f"Scraping-Werkzeug (Firecrawl) wird für '{url}' verwendet")
    api_url = "https://api.firecrawl.dev/v0/scrape"
    headers = {"Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}", "Content-Type": "application/json"}
    payload = {"url": url, "pageOptions": {"onlyMainContent": True}}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(api_url, json=payload, headers=headers, timeout=45)
            response.raise_for_status()
            data = response.json()
            return data.get("data", {}).get("markdown", "") if data.get("success") else ""
        except Exception as e:
            logger.error(f"Fehler im Scraping-Werkzeug für {url}: {e}")
            return ""

# ==============================================================================
# TEIL 3: LOGIK FÜR DEN "DEEP" MODUS - EXPLORATORY GRAPH AGENT
# ==============================================================================

async def build_knowledge_graph_agent(user_query: str, stream_callback: callable) -> Dict[str, Any]:
    """Baut iterativ ein Wissensnetz auf."""
    knowledge_graph = {"nodes": {}, "edges": []}
    
    # Phase 1: Initiale Knotenpunkte identifizieren
    await stream_callback({"type": "status", "content": "Phase 1: Identifiziere Schlüsselkonzepte..."})
    prompt1 = f"""
    Identifiziere für die medizinische Anfrage "{user_query}" die 3-5 zentralen, übergeordneten Konzepte, die für eine tiefgehende Antwort erforderlich sind.
    Gib das Ergebnis ausschließlich als JSON-Liste von Strings zurück. Beispiel: ["Definition und Pathophysiologie", "Diagnostische Verfahren", "Therapeutische Strategien"]
    """
    response1 = await llm_json_model.generate_content_async(prompt1)
    initial_nodes = json.loads(response1.text)
    
    for node_name in initial_nodes:
        knowledge_graph["nodes"][node_name] = {"content": "", "sources": []}

    # Phase 2: Graph iterativ erweitern
    for node_name in initial_nodes:
        await stream_callback({"type": "status", "content": f"Phase 2: Erweitere das Konzept '{node_name}'..."})
        prompt2 = f"""
        Erstelle für das Konzept "{node_name}" im Kontext der Anfrage "{user_query}" eine präzise, zweisprachige Suchanfrage für medizinische Leitlinien und Fachartikel.
        Gib das Ergebnis ausschließlich als JSON-Objekt zurück. Beispiel: {{"query_de": "Lungenembolie Diagnostik Leitlinie", "query_en": "pulmonary embolism diagnosis guideline"}}
        """
        response2 = await llm_json_model.generate_content_async(prompt2)
        queries = json.loads(response2.text)
        
        site_filter = "(site:awmf.org OR site:leitlinien.de OR site:escardio.org OR site:dgk.org OR site:pubmed.ncbi.nlm.nih.gov)"
        search_results = await execute_search_tool(f"{queries['query_de']} OR {queries['query_en']}", site_filter)
        
        if search_results:
            # Scrape der Top-Quelle für dieses Konzept
            top_source_url = search_results[0]['link']
            content = await scrape_tool(top_source_url)
            if content:
                knowledge_graph["nodes"][node_name]["content"] = content
                knowledge_graph["nodes"][node_name]["sources"].append(top_source_url)
    
    return knowledge_graph

async def find_synthesis_path_agent(user_query: str, knowledge_graph: Dict[str, Any]) -> List[str]:
    """Bestimmt den logischsten Pfad durch das Wissensnetz für die Synthese."""
    node_names = list(knowledge_graph["nodes"].keys())
    prompt = f"""
    Gegeben ist die Anfrage "{user_query}" und eine Liste von Wissensknoten.
    Ordne die Knoten in der logischsten Reihenfolge an, um eine kohärente und professionelle Antwort zu strukturieren.
    
    Verfügbare Knoten: {node_names}
    
    Gib ausschließlich eine JSON-Liste der geordneten Knotennamen zurück.
    """
    response = await llm_json_model.generate_content_async(prompt)
    return json.loads(response.text)

async def creative_synthesis_agent(user_query: str, ordered_nodes: List[str], knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
    """Erstellt die finale, kreative JSON-Antwort basierend auf dem geordneten Pfad."""
    context_str = ""
    source_map = {}
    source_counter = 1
    
    for node_name in ordered_nodes:
        node_data = knowledge_graph["nodes"].get(node_name, {})
        if node_data.get("content"):
            # Quellen-Mapping für Zitate erstellen
            citation_indices = []
            for url in node_data["sources"]:
                if url not in source_map:
                    source_map[url] = source_counter
                    source_counter += 1
                citation_indices.append(source_map[url])
            
            citations = "".join([f"<sup>{idx}</sup>" for idx in citation_indices])
            context_str += f"### Konzept: {node_name} {citations}\n\n{node_data['content']}\n\n---\n\n"

    prompt = f"""
    Du bist ein medizinischer Datenanalyst. Deine Aufgabe ist es, die Anfrage eines Arztes zu beantworten, indem du die bereitgestellten Informationen in ein kreatives, strukturiertes JSON-Format umwandelst.

    **Anfrage des Arztes:** "{user_query}"
    **Kontext aus den Wissensknoten:**
    {context_str}

    **Deine Aufgabe:**
    Fülle das folgende JSON-Schema aus. Sei kreativ bei der Wahl der Inhaltsblöcke. Nutze die Zitate aus dem Kontext.
    ```json
    {{
      "reponse_courte": "Eine prägnante Zusammenfassung der Antwort in 1-2 Sätzen.",
      "mots_cles": ["Schlüsselwort 1", "Schlüsselwort 2"],
      "reponse_detaillee": [
        {{
          "type": "paragraphe", "titre": "Titel des Abschnitts", "contenu": "Detaillierter Absatz mit Zitaten<sup>1</sup>."
        }},
        {{
          "type": "tableau", "titre": "Vergleichende Tabelle", "entetes": ["Header 1", "Header 2"], "lignes": [["Zelle 1", "Zelle 2"]]
        }},
        {{
          "type": "diagramme_mermaid", "titre": "Algorithmus", "contenu": "graph TD\\nA --> B;"
        }}
      ]
    }}
    ```
    """
    response = await llm_text_model.generate_content_async(prompt)
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    final_json = json.loads(match.group(0)) if match else {}
    
    # Füge die Quellenliste hinzu
    final_json["sources"] = [{"id": v, "url": k} for k, v in source_map.items()]
    return final_json

# ==============================================================================
# TEIL 6: ORCHESTRIERUNGS-PIPELINE
# ==============================================================================

async def deep_mode_pipeline(query: str, stream_callback: callable):
    """Führt die "Exploratory Graph"-Logik aus."""
    knowledge_graph = await build_knowledge_graph_agent(query, stream_callback)
    
    await stream_callback({"type": "status", "content": "Phase 3: Bestimme den logischen Antwortpfad..."})
    synthesis_path = await find_synthesis_path_agent(query, knowledge_graph)
    
    await stream_callback({"type": "status", "content": "Phase 4: Generiere die finale strukturierte Antwort..."})
    final_json_response = await creative_synthesis_agent(query, synthesis_path, knowledge_graph)
    
    await stream_callback({"type": "final_response", "content": final_json_response})

async def rapid_mode_pipeline(query: str, stream_callback: callable):
    """Führt eine schnellere, direktere Recherche aus."""
    await stream_callback({"type": "status", "content": "Starte schnelle Recherche..."})
    # Vereinfachte Logik für den schnellen Modus (kann hier implementiert werden)
    # Vorerst eine Fehlermeldung, um den Fokus auf den deep mode zu legen
    await asyncio.sleep(1)
    final_json = {"reponse_courte": "Der schnelle Modus ist derzeit in Entwicklung. Bitte verwenden Sie den 'Approfondi'-Modus für eine vollständige Antwort.", "mots_cles": [], "reponse_detaillee": []}
    await stream_callback({"type": "final_response", "content": final_json})


@app.post("/research-stream")
async def perform_research_stream(request: ResearchRequest):
    async def event_generator():
        async def stream_callback(data: Dict):
            yield {"data": json.dumps(data)}
        
        try:
            if request.mode == 'deep':
                await deep_mode_pipeline(request.query, stream_callback)
            else: # rapid
                await rapid_mode_pipeline(request.query, stream_callback)
        except Exception as e:
            logger.exception(f"Ein schwerwiegender Fehler ist in der Pipeline aufgetreten: {e}")
            error_message = json.dumps({"type": "error", "content": f"Ein interner Serverfehler ist aufgetreten: {e}"})
            yield {"data": error_message}

    return EventSourceResponse(event_generator())

# --- Startup und Health Check ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starte Agent mit Exploratory Graph Logik...")
    if not all([settings.GEMINI_API_KEY, settings.SERPER_API_KEY, settings.FIRECRAWL_API_KEY]):
        logger.error("FATAL: Wichtige API-Schlüssel fehlen.")
    else:
        logger.info("Alle API-Schlüssel sind konfiguriert.")

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok", "version": app.version}

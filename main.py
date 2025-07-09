# ==============================================================================
# DATEI: main.py
# ROLLE: Implementiert eine zweigleisige Strategie:
#        - 'rapid' Modus: Ein schneller, dynamischer Plan.
#        - 'deep' Modus: Ein fortschrittlicher "Exploratory Graph"-Agent, der
#          ein Wissensnetz aufbaut und daraus eine tiefgehende Synthese erstellt.
# KORREKTUR: Die asynchrone Logik wurde überarbeitet, um den Fehler
#            "async_generator can't be used in 'await' expression" zu beheben.
# SPRACHE: Deutsch
# VERSION: 7.1.0
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
    version="7.1.0"
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
# TEIL 3: GEMEINSAME AGENTEN-LOGIK
# ==============================================================================

async def generate_dynamic_search_plan(user_query: str, mode: str) -> List[Dict[str, str]]:
    mode_instruction = (
        "Erstelle einen umfassenden, mehrstufigen Rechercheplan mit 4-6 Schritten, um das Thema tiefgehend zu untersuchen. Kombiniere Definitions-, Leitlinien- und Studienrecherchen."
        if mode == "deep"
        else "Erstelle einen kurzen und effizienten Rechercheplan mit 2-3 Schritten, um die Frage direkt zu beantworten."
    )
    prompt = f"""
    Du bist ein medizinischer Recherche-Stratege. Erstelle einen optimalen Rechercheplan für die Anfrage eines Arztes.
    Anfrage: "{user_query}"
    Anweisungen: {mode_instruction}
    Definiere für jeden Schritt ein Werkzeug (`guideline_search`, `academic_search`, `web_search`) und zweisprachige Suchanfragen (`query_de`, `query_en`).
    Gib das Ergebnis ausschließlich als JSON-Array von Objekten zurück.
    """
    response = await llm_json_model.generate_content_async(prompt)
    return json.loads(response.text)

async def filter_and_rank_sources(user_query: str, sources: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if not sources: return []
    num_sources_to_select = 5 if mode == 'deep' else 3
    source_summaries = "\n".join([f"ID: {i}, Titel: {s['title']}, URL: {s['link']}" for i, s in enumerate(sources)])
    prompt = f"""
    Bewerte die Relevanz der folgenden Quellen für die Anfrage: "{user_query}".
    Priorisiere offizielle Leitlinien und systematische Reviews.
    Gib ausschließlich eine JSON-Liste der IDs der {num_sources_to_select} besten Quellen zurück.
    """
    response = await llm_json_model.generate_content_async(prompt)
    best_ids = json.loads(response.text)
    return [sources[i] for i in best_ids if i < len(sources)]

async def creative_synthesis_agent(user_query: str, knowledge_graph: Dict[str, Any]) -> Dict[str, Any]:
    context_str = ""
    source_map = {}
    source_counter = 1
    
    for node_name, node_data in knowledge_graph.items():
        if node_data.get("content"):
            citation_indices = []
            for url in node_data["sources"]:
                if url not in source_map:
                    source_map[url] = source_counter
                    source_counter += 1
                citation_indices.append(source_map[url])
            
            citations = "".join([f"<sup>{idx}</sup>" for idx in citation_indices])
            context_str += f"### Konzept: {node_name} {citations}\n\n{node_data['content']}\n\n---\n\n"

    prompt = f"""
    Du bist ein medizinischer Datenanalyst. Wandle die Anfrage eines Arztes und den bereitgestellten Kontext in ein strukturiertes JSON-Format um.
    **Anfrage:** "{user_query}"
    **Kontext:**
    {context_str}
    **Deine Aufgabe:**
    Fülle das folgende JSON-Schema aus. Sei kreativ bei der Wahl der Inhaltsblöcke. Zitiere Informationen mit hochgestellten Zahlen, z.B. `...Text...<sup>1</sup>`.
    ```json
    {{
      "reponse_courte": "Eine prägnante Zusammenfassung der Antwort in 1-2 Sätzen.",
      "mots_cles": ["Schlüsselwort 1", "Schlüsselwort 2"],
      "reponse_detaillee": [
        {{"type": "paragraphe", "titre": "Titel", "contenu": "Absatz mit Zitaten<sup>1</sup>."}},
        {{"type": "tableau", "titre": "Tabelle", "entetes": ["H1", "H2"], "lignes": [["Z1", "Z2"]]}},
        {{"type": "diagramme_mermaid", "titre": "Diagramm", "contenu": "graph TD\\nA --> B;"}}
      ]
    }}
    ```
    """
    response = await llm_text_model.generate_content_async(prompt)
    match = re.search(r'\{.*\}', response.text, re.DOTALL)
    final_json = json.loads(match.group(0)) if match else {}
    final_json["sources"] = [{"id": v, "url": k} for k, v in source_map.items()]
    return final_json

# ==============================================================================
# TEIL 4: PIPELINES FÜR "DEEP" UND "RAPID" MODUS
# ==============================================================================

async def base_pipeline(query: str, mode: str) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Eine generische Pipeline, die von beiden Modi verwendet wird.
    Die Komplexität wird durch den generierten Plan gesteuert.
    """
    yield {"type": "status", "content": "Entwickle eine Recherchestrategie..."}
    try:
        search_plan = await generate_dynamic_search_plan(query, mode)
    except Exception as e:
        logger.error(f"Fehler bei der Planerstellung: {e}")
        yield {"type": "error", "content": f"Konnte keinen Rechercheplan erstellen: {e}"}
        return

    knowledge_graph = {}
    for step in search_plan:
        yield {"type": "status", "content": f"Schritt {step['step']}: {step['description']}"}
        
        site_filters = {
            "guideline_search": "(site:awmf.org OR site:leitlinien.de OR site:escardio.org OR site:nice.org.uk)",
            "academic_search": "site:pubmed.ncbi.nlm.nih.gov"
        }
        site_filter = site_filters.get(step['tool'], "")
        
        de_task = execute_search_tool(step['query_de'], site_filter)
        en_task = execute_search_tool(step['query_en'], site_filter)
        results = await asyncio.gather(de_task, en_task)
        
        all_step_results = [item for sublist in results for item in sublist]
        
        if all_step_results:
            top_source_url = all_step_results[0]['link']
            content = await scrape_tool(top_source_url)
            if content:
                node_name = step['description']
                knowledge_graph[node_name] = {"content": content, "sources": [top_source_url]}

    if not knowledge_graph:
        yield {"type": "error", "content": "Keine Informationen gefunden oder Inhalte konnten nicht extrahiert werden."}
        return

    yield {"type": "status", "content": "Strukturiere die finale Antwort..."}
    try:
        final_json_response = await creative_synthesis_agent(query, knowledge_graph)
        yield {"type": "final_response", "content": final_json_response}
    except Exception as e:
        logger.error(f"Fehler bei der finalen Synthese: {e}")
        yield {"type": "error", "content": f"Konnte die finale Antwort nicht erstellen: {e}"}

# ==============================================================================
# TEIL 5: API-ENDPUNKTE
# ==============================================================================

@app.post("/research-stream")
async def perform_research_stream(request: ResearchRequest):
    """Wählt die passende Pipeline basierend auf dem Modus."""
    logger.info(f"Rechercheanfrage für '{request.query}' im Modus '{request.mode}' erhalten.")
    
    pipeline_generator = base_pipeline(request.query, request.mode)

    async def event_wrapper():
        """Verpackt die Pipeline-Events für SSE."""
        try:
            async for event_data in pipeline_generator:
                yield {"data": json.dumps(event_data)}
        except Exception as e:
            logger.exception(f"Ein schwerwiegender Fehler ist in der Pipeline aufgetreten: {e}")
            error_message = json.dumps({"type": "error", "content": f"Ein interner Serverfehler ist aufgetreten: {e}"})
            yield {"data": error_message}
            
    return EventSourceResponse(event_wrapper())

@app.on_event("startup")
async def startup_event():
    logger.info("Starte Agent mit dynamischer Pipeline-Logik...")
    if not all([settings.GEMINI_API_KEY, settings.SERPER_API_KEY, settings.FIRECRAWL_API_KEY]):
        logger.error("FATAL: Wichtige API-Schlüssel fehlen.")
    else:
        logger.info("Alle API-Schlüssel sind konfiguriert.")

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok", "version": app.version}

# ==============================================================================
# DATEI: main.py
# ROLLE: Dynamischer KI-Orchestrator, der für jede Anfrage einen maßgeschneiderten,
#        mehrstufigen Rechercheplan erstellt und ausführt.
# SPRACHE: Deutsch
# VERSION: 5.1.0
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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

# ==============================================================================
# TEIL 1: KONFIGURATION UND INITIALISIERUNG
# ==============================================================================

# --- Logging-Konfiguration ---
def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        root_logger.addHandler(stream_handler)

# --- Umgebungsvariablen und Secrets ---
load_dotenv()

class Settings(BaseSettings):
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY")
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY")

    class Config:
        env_file = ".env"

settings = Settings()
setup_logging()
logger = logging.getLogger(__name__)

# --- FastAPI-App-Initialisierung ---
app = FastAPI(
    title="Dynamischer Medizinischer Forschungsagent",
    description="Ein KI-Agent, der für jede Anfrage maßgeschneiderte Recherchestrategien entwickelt und ausführt, um Antworten auf Facharztniveau zu liefern.",
    version="5.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini-Modell-Initialisierung ---
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Google Gemini-Modell initialisiert.")
except Exception as e:
    logger.error(f"Fehler bei der Konfiguration der Gemini-API: {e}")
    llm_model = None

# ==============================================================================
# TEIL 2: DATENMODELLE (PYDANTIC)
# ==============================================================================

class ResearchRequest(BaseModel):
    query: str
    mode: Literal["rapid", "deep"]

class Source(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None

# ==============================================================================
# TEIL 3: SPEZIALISIERTE AGENTEN-WERKZEUGE
# ==============================================================================

async def execute_search_tool(query: str, site_filter: str = "") -> List[Dict[str, Any]]:
    """Ein einziges, flexibles Such-Werkzeug."""
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
    """Robustes Scraping-Werkzeug mit Firecrawl."""
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
# TEIL 4: LOGIK DES DYNAMISCHEN ORCHESTRIERUNGS-AGENTEN
# ==============================================================================

async def generate_dynamic_search_plan(user_query: str, mode: str) -> List[Dict[str, str]]:
    """
    Schritt 1: Das KI-Gehirn. Erstellt einen dynamischen, mehrstufigen Rechercheplan.
    """
    mode_instruction = (
        "Erstelle einen umfassenden, mehrstufigen Rechercheplan mit 4-6 Schritten, um das Thema tiefgehend zu untersuchen. Kombiniere Definitions-, Leitlinien- und Studienrecherchen."
        if mode == "deep"
        else "Erstelle einen kurzen und effizienten Rechercheplan mit 2-3 Schritten, um die Frage direkt zu beantworten."
    )

    prompt = f"""
    Du bist ein medizinischer Recherche-Stratege. Deine Aufgabe ist es, einen optimalen Rechercheplan für die Anfrage eines Arztes zu erstellen.

    Anfrage des Arztes: "{user_query}"

    Anweisungen:
    1.  **Plan erstellen:** {mode_instruction}
    2.  **Werkzeuge definieren:** Wähle für jeden Schritt das passende Werkzeug:
        - `guideline_search`: Für die Suche nach offiziellen Leitlinien.
        - `academic_search`: Für die Suche nach klinischen Studien auf PubMed.
        - `web_search`: Für die Suche nach Übersichtsartikeln, Fachinformationen oder Grundlagen.
    3.  **Zweisprachige Anfragen:** Formuliere für jeden Schritt eine präzise Suchanfrage (`query_de` auf Deutsch und `query_en` auf Englisch).

    Gib das Ergebnis ausschließlich als JSON-Array von Objekten zurück.
    Beispiel für einen Plan:
    [
      {{
        "step": 1,
        "description": "Suche nach der S3-Leitlinie zur Definition und initialen Diagnostik.",
        "tool": "guideline_search",
        "query_de": "AWMF S3-Leitlinie Lungenembolie Definition Diagnostik",
        "query_en": "pulmonary embolism guideline definition diagnosis"
      }},
      {{
        "step": 2,
        "description": "Suche nach aktuellen Meta-Analysen zu neuen Antikoagulanzien.",
        "tool": "academic_search",
        "query_de": "Lungenembolie neue Antikoagulanzien Meta-Analyse",
        "query_en": "pulmonary embolism novel anticoagulants meta-analysis"
      }}
    ]
    """
    try:
        response = await llm_model.generate_content_async(prompt)
        match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Kein JSON-Array in der LLM-Antwort gefunden")
    except Exception as e:
        logger.error(f"Fehler bei der Planerstellung: {e}. Fallback-Plan wird verwendet.")
        return [{"step": 1, "description": "Standard-Websuche", "tool": "web_search", "query_de": user_query, "query_en": user_query}]

async def filter_and_rank_sources(user_query: str, sources: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    """Schritt 3: Filtert und bewertet Quellen nach Relevanz und Autorität."""
    if not sources: return []
    num_sources_to_select = 5 if mode == 'deep' else 3
    source_summaries = "\n".join([f"ID: {i}, Titel: {s['title']}, URL: {s['link']}" for i, s in enumerate(sources)])
    prompt = f"""
    Bewerte die Relevanz der folgenden Quellen für die Anfrage eines Arztes: "{user_query}".
    Priorisiere offizielle Leitlinien (AWMF, ESC), systematische Reviews und Publikationen von Fachgesellschaften.

    Quellen:
    {source_summaries}

    Gib ausschließlich eine JSON-Liste der IDs der {num_sources_to_select} besten Quellen zurück. Beispiel: [0, 2, 5]
    """
    try:
        response = await llm_model.generate_content_async(prompt)
        match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if match:
            best_ids = json.loads(match.group(0))
            return [sources[i] for i in best_ids if i < len(sources)]
        return sources[:num_sources_to_select]
    except Exception:
        return sources[:num_sources_to_select]

async def synthesize_answer_tool(user_query: str, research_data: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Schritt 5: Generiert einen professionellen, zitierten Bericht, der auf die Nutzerfrage zugeschnitten ist."""
    context_str = ""
    for i, d in enumerate(research_data):
        context_str += f"### Quelle {i+1}: {d['title']} (URL: {d['url']})\n\n{d['content']}\n\n---\n\n"

    prompt = f"""
    Du bist ein deutscher Facharzt und Autor von medizinischen Fachartikeln. Deine Aufgabe ist es, einen professionellen und gut ausgearbeiteten Bericht zu verfassen, der die Anfrage eines Kollegen präzise und umfassend beantwortet.

    **Anfrage des Arztes:** "{user_query}"

    **Anweisungen für den Bericht:**
    1.  **Professionelle Struktur:** Gliedere deine Antwort in logische Abschnitte, die zur Frage passen. Eine gute Struktur könnte sein:
        - **Einleitung:** Fasse die Kernfrage kurz zusammen und gib einen Überblick über die Antwort.
        - **Hauptteil:** Behandle die Hauptaspekte der Frage detailliert. Nutze hierfür aussagekräftige Unterüberschriften (z.B. "Pathophysiologie", "Diagnostische Kriterien", "Therapeutische Optionen").
        - **Schlussfolgerung/Zusammenfassung:** Fasse die wichtigsten Punkte am Ende prägnant zusammen.
    2.  **Fokus auf die Frage:** Der gesamte Bericht muss sich darauf konzentrieren, die spezifische Frage des Nutzers zu beantworten. Gehe in die Tiefe, aber bleibe immer relevant. Vermeide kurze, oberflächliche Antworten.
    3.  **Formatierung auf Fachniveau:** Nutze Markdown-Tabellen für Vergleiche (z.B. Medikamentendosierungen, Studienergebnisse), Listen für Symptome oder Kriterien und Codeblöcke ` ``` für Algorithmen oder komplexe Schemata.
    4.  **Zitieren mit Hochzahlen:** Zitiere Informationen direkt im Text mit hochgestellten Zahlen im Markdown-Format, z.B. `...Text...<sup>1</sup>`.
    5.  **Evidenzbasiert und präzise:** Halte dich strikt an die Informationen aus den bereitgestellten Quellen.

    **Kontext aus extrahierten Quellen:**
    {context_str}

    **Dein professioneller, detaillierter und zitierter Bericht im Markdown-Format:**
    """
    try:
        stream = await llm_model.generate_content_async(prompt, stream=True)
        async for chunk in stream:
            yield json.dumps({"type": "chunk", "content": chunk.text})
    except Exception as e:
        logger.exception("Fehler bei der Antwortsynthese.")
        yield json.dumps({"type": "error", "content": f"Fehler bei der Antwort-Erstellung: {e}"})

# ==============================================================================
# TEIL 5: ORCHESTRIERUNGS-PIPELINE
# ==============================================================================

async def research_orchestrator_pipeline(query: str, mode: str) -> AsyncGenerator[str, None]:
    yield json.dumps({"type": "status", "content": "Entwickle eine Recherchestrategie..."})
    search_plan = await generate_dynamic_search_plan(query, mode)
    
    all_results = []
    for step in search_plan:
        yield json.dumps({"type": "status", "content": f"Schritt {step['step']}: {step['description']}"})
        
        site_filters = {
            "guideline_search": "(site:awmf.org OR site:leitlinien.de OR site:escardio.org OR site:nice.org.uk)",
            "academic_search": "site:pubmed.ncbi.nlm.nih.gov"
        }
        site_filter = site_filters.get(step['tool'], "")
        
        # Parallele Suche in Deutsch und Englisch
        de_task = execute_search_tool(step['query_de'], site_filter)
        en_task = execute_search_tool(step['query_en'], site_filter)
        results = await asyncio.gather(de_task, en_task)
        all_results.extend([item for sublist in results for item in sublist])

    if not all_results:
        yield json.dumps({"type": "error", "content": "Keine Dokumente gefunden."}); return

    unique_results = list({item['link']: item for item in all_results}.values())
    yield json.dumps({"type": "status", "content": f"{len(unique_results)} potenzielle Quellen gefunden. Bewerte Relevanz..."})

    best_sources_metadata = await filter_and_rank_sources(query, unique_results, mode)
    if not best_sources_metadata:
        yield json.dumps({"type": "error", "content": "Keine relevanten Quellen bestimmbar."}); return
        
    yield json.dumps({"type": "status", "content": f"Extrahiere Inhalte aus {len(best_sources_metadata)} Schlüsselquellen..."})

    scrape_tasks = [scrape_tool(src['link']) for src in best_sources_metadata]
    scraped_contents = await asyncio.gather(*scrape_tasks)

    research_data, final_sources = [], []
    for i, content in enumerate(scraped_contents):
        if content and len(content) > 100:
            metadata = best_sources_metadata[i]
            research_data.append({"url": metadata['link'], "title": metadata['title'], "content": content})
            final_sources.append(Source(title=metadata['title'], url=metadata['link'], snippet=metadata.get('snippet')).model_dump())

    if not research_data:
        yield json.dumps({"type": "error", "content": "Inhalte konnten nicht extrahiert werden."}); return

    yield json.dumps({"type": "status", "content": "Erstelle evidenzbasierte Antwort..."})
    async for chunk in synthesize_answer_tool(query, research_data):
        yield chunk
        
    yield json.dumps({"type": "sources", "content": final_sources})

# ==============================================================================
# TEIL 6: API-ENDPUNKTE
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Starte dynamischen Recherche-Agenten...")
    if not all([settings.GEMINI_API_KEY, settings.SERPER_API_KEY, settings.FIRECRAWL_API_KEY]):
        logger.error("FATAL: Wichtige API-Schlüssel fehlen.")
    else:
        logger.info("Alle API-Schlüssel sind konfiguriert.")

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok", "version": app.version}

@app.post("/research-stream")
async def perform_research_stream(request: ResearchRequest):
    logger.info(f"Rechercheanfrage für '{request.query}' im Modus '{request.mode}' erhalten.")
    return EventSourceResponse(research_orchestrator_pipeline(request.query, request.mode))

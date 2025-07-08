# ==============================================================================
# DATEI: main.py
# ROLLE: Hochmoderner medizinischer Forschungs-Orchestrator, der die Absicht
#        der Anfrage analysiert und dynamische, auf Leitlinien fokussierte
#        Tools verwendet, um Antworten auf Facharztniveau zu liefern.
# SPRACHE: Deutsch
# VERSION: 4.0.0
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
    title="Medizinischer Forschungsagent für Fachärzte",
    description="Ein fortschrittlicher KI-Agent, der die Absicht von Anfragen versteht und dynamische, auf Leitlinien fokussierte Recherche-Tools für präzise Antworten nutzt.",
    version="4.0.0"
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

async def web_search_tool(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Allgemeines Websuche-Werkzeug."""
    logger.info(f"Allgemeines Websuche-Werkzeug wird für '{query}' verwendet")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {'X-API-KEY': settings.SERPER_API_KEY, 'Content-Type': 'application/json'}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json().get('organic', [])
        except Exception as e:
            logger.error(f"Fehler im Websuche-Werkzeug: {e}")
            return []

async def guideline_search_tool(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """
    Spezialisiertes Werkzeug zur Suche nach medizinischen Leitlinien auf maßgeblichen deutschen und internationalen Portalen.
    """
    logger.info(f"Leitlinien-Suche-Werkzeug wird für '{query}' verwendet")
    # Fokus auf deutsche und wichtige internationale Leitlinien-Portale
    guideline_query = f"({query}) (site:awmf.org OR site:leitlinien.de OR site:kbv.de OR site:escardio.org OR site:nice.org.uk OR site:dgk.org)"
    return await web_search_tool(guideline_query, num_results)

async def academic_search_tool(query: str, num_results: int = 3) -> List[Dict[str, Any]]:
    """Akademisches Such-Werkzeug mit Fokus auf PubMed."""
    logger.info(f"Akademisches Such-Werkzeug wird für '{query}' verwendet")
    pubmed_query = f"site:pubmed.ncbi.nlm.nih.gov {query}"
    return await web_search_tool(pubmed_query, num_results)

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
            if data.get("success"):
                return data.get("data", {}).get("markdown", "")
            return ""
        except Exception as e:
            logger.error(f"Fehler im Scraping-Werkzeug für {url}: {e}")
            return ""

# ==============================================================================
# TEIL 4: LOGIK DES ORCHESTRIERUNGS-AGENTEN
# ==============================================================================

async def analyze_query_intent(user_query: str) -> Dict[str, Any]:
    """
    Schritt 1: Analysiert die Anfrage, um die Absicht und die passenden Suchanfragen zu ermitteln.
    Fokus liegt auf der Unterscheidung zwischen Grundlagenwissen und spezifischen klinischen Fragen.
    """
    prompt = f"""
    Analysiere die folgende medizinische Anfrage eines Arztes: "{user_query}"

    Bestimme die primäre Absicht. Die Optionen sind:
    - 'definition_grundlagen': Der Arzt bittet um eine Definition, eine Erklärung von Pathophysiologie, Symptomen oder Grundlagen.
    - 'diagnostik_therapie': Die Anfrage bezieht sich auf Diagnostik, Behandlungsstrategien, Dosierungen oder Leitlinienempfehlungen.
    - 'studien_daten': Der Arzt sucht nach spezifischen klinischen Studiendaten, Meta-Analysen oder Evidenz.

    Generiere basierend auf der Absicht drei Arten von Suchanfragen auf Englisch:
    1.  `guideline_query`: Eine Anfrage, die speziell nach offiziellen Leitlinien sucht (z.B. "AWMF S3-Leitlinie Lungenembolie").
    2.  `general_query`: Eine breitere Anfrage für ein allgemeines Verständnis (z.B. "pulmonary embolism diagnosis and management").
    3.  `academic_query`: Eine technische Anfrage für PubMed (z.B. "pulmonary embolism treatment new anticoagulants trial").

    Gib das Ergebnis ausschließlich als JSON-Objekt zurück.
    Beispiel:
    {{
      "intent": "diagnostik_therapie",
      "guideline_query": "AWMF S3-Leitlinie Lungenembolie Diagnostik",
      "general_query": "pulmonary embolism diagnostic algorithm",
      "academic_query": "pulmonary embolism d-dimer sensitivity specificity meta-analysis"
    }}
    """
    try:
        response = await llm_model.generate_content_async(prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Kein JSON in der LLM-Antwort gefunden")
    except Exception as e:
        logger.error(f"Fehler bei der Absichtsanalyse: {e}. Fallback wird verwendet.")
        return {
            "intent": "definition_grundlagen",
            "guideline_query": user_query + " Leitlinie",
            "general_query": user_query,
            "academic_query": user_query
        }

async def filter_and_rank_sources(user_query: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Schritt 3: Filtert und bewertet Quellen, um die relevantesten und autoritativsten zu priorisieren.
    """
    if not sources:
        return []

    source_summaries = "\n".join([f"ID: {i}, Titel: {s['title']}, URL: {s['link']}, Snippet: {s.get('snippet', '')}" for i, s in enumerate(sources)])
    prompt = f"""
    Bewerte angesichts der Benutzeranfrage und einer Liste von Quellen die 3-5 relevantesten Quellen.
    Priorisiere Quellen von bekannten medizinischen Fachgesellschaften (AWMF, DGK, ESC) und offiziellen Leitlinien-Portalen.
    Bevorzuge systematische Reviews und Leitlinien gegenüber einzelnen Fallberichten für allgemeine Fragen.

    Anfrage: "{user_query}"

    Quellen:
    {source_summaries}

    Gib ausschließlich eine JSON-Liste der IDs der besten Quellen zurück. Beispiel: [0, 2, 5]
    """
    try:
        response = await llm_model.generate_content_async(prompt)
        match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if match:
            best_ids = json.loads(match.group(0))
            return [sources[i] for i in best_ids if i < len(sources)]
        return sources[:3]
    except Exception as e:
        logger.error(f"Fehler beim Filtern der Quellen: {e}. Die ersten 3 Quellen werden verwendet.")
        return sources[:3]

async def synthesize_report_tool(user_query: str, research_data: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """
    Schritt 5: Generiert die finale Synthese. Integriert Zitate und nutzt komplexe Markdown-Formatierungen.
    """
    context_str = ""
    for i, d in enumerate(research_data):
        context_str += f"### Quelle {i+1}: {d['title']} (URL: {d['url']})\n\n{d['content']}\n\n---\n\n"

    prompt = f"""
    Du bist ein deutscher Facharzt und verfasst eine präzise, evidenzbasierte Zusammenfassung für einen Kollegen.
    Analysiere den folgenden Kontext, der aus mehreren medizinischen Quellen extrahiert wurde, um die Anfrage des Benutzers zu beantworten.

    **Anfrage des Arztes:** "{user_query}"

    **Deine Aufgabe:**
    1.  **Strukturierter Bericht:** Erstelle einen Bericht mit klaren Abschnitten (z.B. Einleitung, Diagnostik, Therapie, Prognose).
    2.  **In-Text-Zitate:** Zitiere Informationen direkt im Text unter Verwendung des Formats `[Quelle X]`, wobei X die Nummer der Quelle aus dem Kontext ist.
    3.  **Reichhaltiges Markdown:** Nutze Markdown-Tabellen zur Darstellung von Vergleichen (z.B. Medikamentendosierungen, Studienergebnisse), Listen für Symptome oder Kriterien und Codeblöcke ` ``` für Algorithmen oder komplexe Dosierungsschemata.
    4.  **Evidenzbasierung:** Halte dich strikt an die Informationen aus den bereitgestellten Quellen. Spekuliere nicht. Wenn Informationen fehlen, weise darauf hin.
    5.  **Sprache:** Verfasse den gesamten Bericht auf Deutsch auf professionellem medizinischem Niveau.

    **Kontext aus extrahierten Quellen:**
    {context_str}

    **Dein detaillierter, zitierter Bericht im Markdown-Format:**
    """
    try:
        stream = await llm_model.generate_content_async(prompt, stream=True)
        async for chunk in stream:
            yield json.dumps({"type": "chunk", "content": chunk.text})
    except Exception as e:
        logger.exception("Fehler bei der Berichtssynthese.")
        yield json.dumps({"type": "error", "content": f"Fehler bei der Berichtserstellung: {e}"})

# ==============================================================================
# TEIL 5: ORCHESTRIERUNGS-PIPELINE
# ==============================================================================

async def research_orchestrator_pipeline(query: str, mode: str) -> AsyncGenerator[str, None]:
    """
    Die vollständige Pipeline, die die Recherche von der Analyse bis zur Synthese orchestriert.
    """
    yield json.dumps({"type": "status", "content": "Analysiere ärztliche Anfrage..."})
    intent_analysis = await analyze_query_intent(query)
    intent = intent_analysis['intent']
    yield json.dumps({"type": "status", "content": f"Absicht erkannt: {intent}. Starte Recherche..."})

    search_tasks = [
        guideline_search_tool(intent_analysis['guideline_query'], num_results=5),
        web_search_tool(intent_analysis['general_query'], num_results=3)
    ]
    if mode == 'deep' or intent == 'studien_daten':
        search_tasks.append(academic_search_tool(intent_analysis['academic_query'], num_results=5))
    
    search_results_lists = await asyncio.gather(*search_tasks)
    all_results = [item for sublist in search_results_lists for item in sublist]

    if not all_results:
        yield json.dumps({"type": "error", "content": "Keine Dokumente für diese Anfrage gefunden."})
        return

    unique_results = list({item['link']: item for item in all_results}.values())
    yield json.dumps({"type": "status", "content": f"{len(unique_results)} potenzielle Quellen gefunden."})

    yield json.dumps({"type": "status", "content": "Bewerte und filtere Quellen nach Relevanz..."})
    best_sources_metadata = await filter_and_rank_sources(query, unique_results)
    
    if not best_sources_metadata:
        yield json.dumps({"type": "error", "content": "Relevante Quellen konnten nicht bestimmt werden."})
        return
        
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
        yield json.dumps({"type": "error", "content": "Inhalte der Schlüsselquellen konnten nicht extrahiert werden."})
        return

    yield json.dumps({"type": "status", "content": "Erstelle evidenzbasierte Synthese..."})
    async for chunk in synthesize_report_tool(query, research_data):
        yield chunk
        
    yield json.dumps({"type": "sources", "content": final_sources})

# ==============================================================================
# TEIL 6: API-ENDPUNKTE
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Starte Orchestrator-Anwendung...")
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

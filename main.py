# ==============================================================================
# DATEI 1: main.py
# ROLLE: Hochentwickelter medizinischer Forschungsagent mit spezialisierten
#        Tools und Echtzeit-Status-Streaming (Server-Sent Events).
# SPRACHE: Deutsch
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
from bs4 import BeautifulSoup

# ==============================================================================
# TEIL 1: KONFIGURATION UND INITIALISIERUNG
# ==============================================================================

# --- Logging-Konfiguration ---
def setup_logging():
    """Konfiguriert das zentrale Logging für die Anwendung."""
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    # Sicherstellen, dass keine doppelten Handler hinzugefügt werden
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        root_logger.addHandler(stream_handler)

# --- Umgebungsvariablen und Secrets ---
load_dotenv()

class Settings(BaseSettings):
    """Lädt Konfigurationen und API-Schlüssel aus Umgebungsvariablen."""
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY") # Für Google Scholar / Allgemeine Suche

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
setup_logging()
logger = logging.getLogger(__name__)

# --- FastAPI-App-Initialisierung ---
app = FastAPI(
    title="Fortschrittlicher Medizinischer KI-Forschungsagent",
    description="Ein KI-Agent, der spezialisierte Tools zur Durchführung von medizinischen Recherchen nutzt und Ergebnisse in Echtzeit streamt.",
    version="2.0.0"
)

# --- CORS-Middleware-Konfiguration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Für die Produktion auf die Frontend-URL beschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Gemini-Modell-Initialisierung ---
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Google Gemini-Modell erfolgreich initialisiert.")
except Exception as e:
    logger.error(f"Fehler bei der Konfiguration der Gemini-API: {e}")
    llm_model = None

# ==============================================================================
# TEIL 2: DATENMODELLE (PYDANTIC)
# ==============================================================================

class ResearchRequest(BaseModel):
    """Modell für die eingehende Rechercheanfrage des Benutzers."""
    query: str = Field(..., description="Die medizinische Frage des Benutzers.")
    mode: Literal["rapid", "deep"] = Field(..., description="Der zu verwendende Recherchemodus: 'schnell' oder 'tiefgehend'.")

class Source(BaseModel):
    """Modell für eine einzelne Informationsquelle."""
    title: str
    url: str
    snippet: Optional[str] = None

class FinalResponse(BaseModel):
    """Modell für die endgültige, vollständige Antwort."""
    summary: str
    sources: List[Source]

# ==============================================================================
# TEIL 3: SPEZIALISIERTE RECHERCHE-TOOLS
# ==============================================================================

async def stream_log(message: str) -> AsyncGenerator[str, None]:
    """Sendet eine Log-Nachricht an den Client über SSE."""
    logger.info(message)
    yield json.dumps({"type": "status", "content": message})

async def search_google_scholar(query: str, num_results: int = 3) -> List[Dict[str, Any]]:
    """
    Führt eine Suche auf Google Scholar über die Serper-API durch.
    """
    logger.info(f"Führe Google Scholar-Suche durch für: '{query}'")
    url = "https://google.serper.dev/scholar"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {'X-API-KEY': settings.SERPER_API_KEY, 'Content-Type': 'application/json'}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            results = response.json()
            return results.get('scholar', [])
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP-Fehler bei Google Scholar-Suche: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            logger.exception("Unerwarteter Fehler bei der Google Scholar-Suche.")
            return []

async def search_pubmed(query: str, num_results: int = 3) -> List[Dict[str, Any]]:
    """
    Simuliert eine Suche auf PubMed, indem eine gezielte Google-Suche verwendet wird.
    Eine direkte PubMed-API (z.B. Entrez) wäre eine robustere Alternative.
    """
    logger.info(f"Führe PubMed-Suche durch für: '{query}'")
    pubmed_query = f"site:pubmed.ncbi.nlm.nih.gov {query}"
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": pubmed_query, "num": num_results})
    headers = {'X-API-KEY': settings.SERPER_API_KEY, 'Content-Type': 'application/json'}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            results = response.json()
            return results.get('organic', [])
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP-Fehler bei PubMed-Suche: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            logger.exception("Unerwarteter Fehler bei der PubMed-Suche.")
            return []

async def scrape_url_content(url: str) -> str:
    """
    Extrahiert den reinen Textinhalt von einer Webseite mit BeautifulSoup.
    """
    logger.info(f"Scraping der URL: {url}")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    async with httpx.AsyncClient(follow_redirects=True) as client:
        try:
            response = await client.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Entfernt Skript- und Stil-Elemente
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            
            # Holt den Text und bereinigt ihn
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            logger.info(f"Scraping von {url} erfolgreich. Länge: {len(text)} Zeichen.")
            return text[:15000] # Begrenzt die Inhaltslänge
        except Exception as e:
            logger.error(f"Fehler beim Scrapen von {url}: {e}")
            return ""

# ==============================================================================
# TEIL 4: KI-LOGIK UND SYNTHESE
# ==============================================================================

async def get_refined_queries(user_query: str) -> Dict[str, str]:
    """
    Generiert optimierte Suchanfragen für verschiedene Datenbanken.
    """
    prompt = f"""
    Basierend auf der medizinischen Anfrage eines Benutzers, erstelle optimierte, englische Suchanfragen für die folgenden Datenbanken:
    1.  **PubMed**: Eine präzise, fachliche Anfrage, die sich auf klinische Studien, Reviews oder Meta-Analysen konzentriert. Verwende ggf. MeSH-ähnliche Begriffe.
    2.  **Google Scholar**: Eine etwas breitere Anfrage, die auch nach Artikeln, Leitlinien und Zitationen sucht.

    Benutzeranfrage: "{user_query}"

    Gib die Antwort ausschließlich als JSON-Objekt im folgenden Format zurück:
    {{
      "pubmed_query": "...",
      "scholar_query": "..."
    }}
    """
    try:
        response = await llm_model.generate_content_async(prompt)
        # Bereinigen und Parsen der JSON-Antwort
        json_str = re.search(r'```json\s*([\s\S]*?)\s*```', response.text)
        if json_str:
            return json.loads(json_str.group(1))
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Fehler beim Verfeinern der Suchanfragen: {e}. Fallback wird verwendet.")
        return {"pubmed_query": user_query, "scholar_query": user_query}

async def synthesize_final_report(user_query: str, research_data: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """
    Erstellt einen detaillierten, strukturierten Bericht aus den gesammelten Daten und streamt ihn.
    """
    context_str = "\n\n---\n\n".join(
        [f"Quelle URL: {d['url']}\n\nTitel: {d['title']}\n\nInhalt:\n{d['content']}" for d in research_data]
    )
    
    prompt = f"""
    Du bist ein Experte für medizinische Forschung und verfasst detaillierte Synthesen für Ärzte.
    Basierend AUSSCHLIESSLICH auf den untenstehenden, extrahierten Inhalten, erstelle einen umfassenden und gut strukturierten Bericht, der die folgende Frage beantwortet.

    **Anfrage des Benutzers:**
    {user_query}

    **Struktur des Berichts:**
    1.  **Zusammenfassung (Executive Summary):** Gib eine kurze, prägnante Antwort auf die Kernfrage.
    2.  **Detaillierte Ergebnisse:** Fasse die wichtigsten Erkenntnisse aus den Quellen zusammen. Wenn möglich, erstelle eine Markdown-Tabelle, um Studien, Dosierungen, Ergebnisse oder andere relevante Daten zu vergleichen.
    3.  **Diskussion und Limitationen:** Erwähne kurz, wenn es widersprüchliche Daten gibt oder welche Limitationen in den Quellen genannt werden.
    4.  **Schlussfolgerung:** Formuliere eine abschließende Schlussfolgerung.

    **WICHTIG:**
    -   Formatiere deine Antwort durchgehend in Markdown.
    -   Sei objektiv und halte dich strikt an die bereitgestellten Informationen.
    -   Erfinde keine Informationen hinzu. Wenn die Daten fehlen, erwähne dies.

    **Gesammelte Daten:**
    {context_str}

    **Dein Bericht:**
    """
    try:
        stream = await llm_model.generate_content_async(prompt, stream=True)
        async for chunk in stream:
            yield json.dumps({"type": "chunk", "content": chunk.text})
    except Exception as e:
        logger.exception("Fehler während der finalen Synthese.")
        yield json.dumps({"type": "error", "content": f"Fehler bei der Berichtserstellung: {e}"})

# ==============================================================================
# TEIL 5: RECHERCHE-PIPELINE (STREAMING)
# ==============================================================================

async def research_pipeline(query: str, mode: str) -> AsyncGenerator[str, None]:
    """
    Die zentrale Pipeline, die den gesamten Rechercheprozess steuert und streamt.
    """
    yield json.dumps({"type": "status", "content": "Anfrage wird analysiert..."})
    
    # Schritt 1: Suchanfragen verfeinern
    refined_queries = await get_refined_queries(query)
    yield json.dumps({"type": "status", "content": f"Spezialisierte Suchanfragen erstellt (PubMed: '{refined_queries['pubmed_query']}')"})
    
    # Schritt 2: Parallele Suche
    yield json.dumps({"type": "status", "content": "Suche in medizinischen Datenbanken wird gestartet..."})
    
    num_results = 5 if mode == 'deep' else 2
    pubmed_task = asyncio.create_task(search_pubmed(refined_queries['pubmed_query'], num_results))
    scholar_task = asyncio.create_task(search_google_scholar(refined_queries['scholar_query'], num_results))
    
    pubmed_results, scholar_results = await asyncio.gather(pubmed_task, scholar_task)
    
    all_results = pubmed_results + scholar_results
    if not all_results:
        yield json.dumps({"type": "error", "content": "Keine relevanten Dokumente in den Datenbanken gefunden."})
        return

    # Eindeutige URLs und Titel sammeln
    unique_sources = {}
    for res in all_results:
        if res.get('link'):
            unique_sources[res['link']] = Source(
                title=res.get('title', 'Unbekannter Titel'),
                url=res['link'],
                snippet=res.get('snippet')
            )
    
    sources_list = list(unique_sources.values())
    yield json.dumps({"type": "status", "content": f"{len(sources_list)} einzigartige Quellen gefunden."})
    
    # Schritt 3: Paralleles Scraping
    yield json.dumps({"type": "status", "content": "Extrahiere Inhalte der gefundenen Quellen..."})
    scrape_tasks = [asyncio.create_task(scrape_url_content(src.url)) for src in sources_list]
    scraped_contents = await asyncio.gather(*scrape_tasks)
    
    # Schritt 4: Daten für die Synthese vorbereiten
    research_data = []
    valid_sources = []
    for i, content in enumerate(scraped_contents):
        if content and len(content) > 100: # Nur Inhalte mit einer Mindestlänge berücksichtigen
            research_data.append({
                "url": sources_list[i].url,
                "title": sources_list[i].title,
                "content": content
            })
            valid_sources.append(sources_list[i].model_dump())

    if not research_data:
        yield json.dumps({"type": "error", "content": "Inhalte der gefundenen Quellen konnten nicht extrahiert werden."})
        return
        
    yield json.dumps({"type": "status", "content": f"Inhalte von {len(research_data)} Quellen werden analysiert. Synthese wird gestartet..."})

    # Schritt 5: Finale Synthese und Streaming der Antwort
    async for chunk in synthesize_final_report(query, research_data):
        yield chunk
        
    # Schritt 6: Quellen senden
    yield json.dumps({"type": "sources", "content": valid_sources})


# ==============================================================================
# TEIL 6: API-ENDPUNKTE
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """Wird beim Start der Anwendung ausgeführt."""
    logger.info("Anwendung startet...")
    if not all([settings.GEMINI_API_KEY, settings.SERPER_API_KEY]):
        logger.error("FATAL: Wichtige API-Schlüssel fehlen. Die Anwendung wird nicht korrekt funktionieren.")
    else:
        logger.info("Alle erforderlichen API-Schlüssel sind konfiguriert.")

@app.get("/health", status_code=200, tags=["System"])
async def health_check():
    """Gesundheits-Check-Endpunkt für Monitoring-Dienste."""
    return {"status": "ok", "version": app.version}

@app.post("/research", tags=["Veraltet"])
async def perform_research_legacy(request: ResearchRequest):
    """Veralteter Endpunkt. Bitte /research-stream verwenden."""
    raise HTTPException(status_code=410, detail="Dieser Endpunkt ist veraltet. Bitte verwenden Sie den '/research-stream'-Endpunkt.")


@app.post("/research-stream", tags=["Recherche"])
async def perform_research_stream(request: ResearchRequest):
    """
    Startet eine Recherche und streamt den Fortschritt und die Ergebnisse
    über Server-Sent Events (SSE).
    """
    logger.info(f"Streaming-Rechercheanfrage erhalten für '{request.query}' im Modus '{request.mode}'")
    
    async def event_generator():
        try:
            async for data_str in research_pipeline(request.query, request.mode):
                yield {"data": data_str}
        except Exception as e:
            logger.exception(f"Ein Fehler ist in der research_pipeline aufgetreten: {e}")
            error_message = json.dumps({"type": "error", "content": f"Ein interner Serverfehler ist aufgetreten: {e}"})
            yield {"data": error_message}

    return EventSourceResponse(event_generator())


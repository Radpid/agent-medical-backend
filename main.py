# ==============================================================================
# DATEI: main.py
# ROLLE: KI-Orchestrator, der einen dynamischen Rechercheplan erstellt und
#        als Endergebnis ein einziges, strukturiertes JSON-Objekt zur
#        kreativen Darstellung im Frontend liefert.
# SPRACHE: Deutsch
# VERSION: 6.0.0
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
    title="Dynamischer Medizinischer Forschungsagent (JSON-API)",
    description="Ein KI-Agent, der maßgeschneiderte Recherchestrategien entwickelt und die Ergebnisse als strukturiertes JSON für eine reichhaltige Frontend-Darstellung liefert.",
    version="6.0.0"
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
    json_generation_config = genai.GenerationConfig(response_mime_type="application/json")
    llm_json_model = genai.GenerativeModel(
        'gemini-1.5-flash',
        generation_config=json_generation_config
    )
    # Standardmodell für Nicht-JSON-Aufgaben
    llm_text_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Google Gemini-Modelle initialisiert.")
except Exception as e:
    logger.error(f"Fehler bei der Konfiguration der Gemini-API: {e}")
    llm_json_model = None
    llm_text_model = None

# ==============================================================================
# TEIL 2: DATENMODELLE UND WERKZEUGE
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
# TEIL 4: LOGIK DES DYNAMISCHEN ORCHESTRIERUNGS-AGENTEN
# ==============================================================================

async def generate_dynamic_search_plan(user_query: str, mode: str) -> List[Dict[str, str]]:
    prompt = f"""
    Du bist ein medizinischer Recherche-Stratege. Erstelle einen optimalen Rechercheplan für die Anfrage eines Arztes.
    Anfrage: "{user_query}"
    Modus: "{mode}" (deep = umfassend, 4-6 Schritte; rapid = effizient, 2-3 Schritte)
    
    Definiere für jeden Schritt ein Werkzeug (`guideline_search`, `academic_search`, `web_search`) und zweisprachige Suchanfragen (`query_de`, `query_en`).
    Gib das Ergebnis ausschließlich als JSON-Array von Objekten zurück.
    """
    try:
        response = await llm_json_model.generate_content_async(prompt)
        # Da wir JSON-Ausgabe erwarten, können wir direkt parsen
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Fehler bei der Planerstellung: {e}. Fallback-Plan wird verwendet.")
        return [{"step": 1, "description": "Standard-Websuche", "tool": "web_search", "query_de": user_query, "query_en": user_query}]

async def filter_and_rank_sources(user_query: str, sources: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if not sources: return []
    num_sources_to_select = 5 if mode == 'deep' else 3
    source_summaries = "\n".join([f"ID: {i}, Titel: {s['title']}, URL: {s['link']}" for i, s in enumerate(sources)])
    prompt = f"""
    Bewerte die Relevanz der folgenden Quellen für die Anfrage: "{user_query}".
    Priorisiere offizielle Leitlinien und systematische Reviews.
    Gib ausschließlich eine JSON-Liste der IDs der {num_sources_to_select} besten Quellen zurück. Beispiel: [0, 2, 5]
    """
    try:
        response = await llm_json_model.generate_content_async(prompt)
        best_ids = json.loads(response.text)
        return [sources[i] for i in best_ids if i < len(sources)]
    except Exception:
        return sources[:num_sources_to_select]

async def synthesize_json_response(user_query: str, research_data: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Schritt 5: Generiert die finale, strukturierte JSON-Antwort.
    """
    context_str = ""
    for i, d in enumerate(research_data):
        context_str += f"### Quelle {i+1}: {d['title']} (URL: {d['url']})\n\n{d['content']}\n\n---\n\n"

    prompt = f"""
    Du bist ein medizinischer Datenanalyst. Deine Aufgabe ist es, die Anfrage eines Arztes zu beantworten, indem du die bereitgestellten Informationen in ein strukturiertes JSON-Format umwandelst.
    Sei kreativ bei der Wahl der Inhaltsblöcke, um die Informationen bestmöglich darzustellen.

    **Anfrage des Arztes:** "{user_query}"

    **Kontext aus extrahierten Quellen:**
    {context_str}

    **Deine Aufgabe:**
    Fülle das folgende JSON-Schema aus. Zitiere Informationen im Text mit hochgestellten Zahlen, z.B. `...Text...<sup>1</sup>`.

    ```json
    {{
      "reponse_courte": "Eine prägnante Zusammenfassung der Antwort in 1-2 Sätzen.",
      "mots_cles": ["Schlüsselwort 1", "Schlüsselwort 2", "Schlüsselwort 3"],
      "reponse_detaillee": [
        {{
          "type": "paragraphe",
          "titre": "Einleitung und Definition",
          "contenu": "Ein detaillierter Absatz, der die Grundlagen erklärt. Zitate wie dieses<sup>1</sup> sind wichtig."
        }},
        {{
          "type": "liste",
          "titre": "Symptome",
          "items": [
            "Symptom A<sup>2</sup>",
            "Symptom B<sup>1,3</sup>",
            "Symptom C"
          ]
        }},
        {{
          "type": "tableau",
          "titre": "Therapieoptionen im Vergleich",
          "entetes": ["Medikament", "Dosierung", "Evidenzgrad<sup>4</sup>"],
          "lignes": [
            ["Medikament X", "10mg täglich", "A"],
            ["Medikament Y", "5mg zweimal täglich", "B"]
          ]
        }},
        {{
          "type": "diagramme_mermaid",
          "titre": "Diagnostischer Algorithmus",
          "contenu": "graph TD\\nA[Verdacht] --> B{{D-Dimer}};\\nB -->|Positiv| C[CT-Angio];\\nB -->|Negativ| D[LAE unwahrscheinlich];"
        }}
      ]
    }}
    ```
    """
    try:
        # Hier verwenden wir das Textmodell, da der Prompt komplex ist und die JSON-Struktur enthält
        response = await llm_text_model.generate_content_async(prompt)
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Kein JSON in der LLM-Antwort gefunden.")
    except Exception as e:
        logger.exception("Fehler bei der JSON-Antwortsynthese.")
        return {"error": f"Fehler bei der Antwort-Erstellung: {e}"}

# ==============================================================================
# TEIL 5: ORCHESTRIERUNGS-PIPELINE
# ==============================================================================

async def research_orchestrator_pipeline(query: str, mode: str) -> AsyncGenerator[str, None]:
    """
    Die vollständige Pipeline, die den gesamten Rechercheprozess steuert und streamt.
    """
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

    yield json.dumps({"type": "status", "content": "Strukturiere die finale Antwort..."})
    final_json_response = await synthesize_json_response(query, research_data)
    
    # Senden der finalen JSON-Antwort und der Quellen in zwei getrennten Events
    yield json.dumps({"type": "final_response", "content": final_json_response})
    yield json.dumps({"type": "sources", "content": final_sources})

# ==============================================================================
# TEIL 6: API-ENDPUNKTE
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Starte dynamischen Recherche-Agenten (JSON-Modus)...")
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

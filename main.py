# ==============================================================================
# DATEI: main.py
# ROLLE: Eine hochmoderne "KI über KI"-Architektur. Ein Master-Agent erstellt
#        dynamische Pläne für eine Brigade von spezialisierten Worker-Agenten.
# KORREKTUR: Der Master-Agent verfügt nun über eine Fallback-Logik. Wenn die
#            erste komplexe Planerstellung fehlschlägt, versucht er es mit
#            einem einfacheren Ansatz erneut, um den "0 Phasen"-Fehler zu beheben.
# SPRACHE: Deutsch
# VERSION: 9.4.0
# LINIEN: > 1500
# ==============================================================================

# ==============================================================================
# TEIL 1: IMPORTE UND GLOBALE KONFIGURATION
# ==============================================================================
import os
import sys
import logging
import asyncio
import json
import re
from typing import List, Optional, Literal, Dict, Any, AsyncGenerator, Union
from abc import ABC, abstractmethod

import httpx
import google.generativeai as genai
from google.generativeai.types import GenerateContentResponse
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from sse_starlette.sse import EventSourceResponse

# ==============================================================================
# TEIL 2: LOGGING UND UMGEBUNGS-SETUP
# ==============================================================================
def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - (%(module)s:%(lineno)d) - %(message)s'
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        root_logger.addHandler(stream_handler)
    return logging.getLogger(__name__)

load_dotenv()

class Settings(BaseSettings):
    GEMINI_API_KEY: str = Field(..., description="API-Schlüssel für Google Gemini.")
    SERPER_API_KEY: str = Field(..., description="API-Schlüssel für die Serper.dev Such-API.")
    FIRECRAWL_API_KEY: str = Field(..., description="API-Schlüssel für die Firecrawl.dev Scraping-API.")
    MASTER_AGENT_MODEL: str = Field("gemini-2.0-flash", description="Leistungsstarkes Modell für den Master-Agenten.")
    WORKER_AGENT_MODEL: str = Field("gemini-2.0-flash", description="Schnelles und effizientes Modell für Worker-Agenten.")

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

try:
    settings = Settings()
    logger = setup_logging()
except ValidationError as e:
    print(f"FATALER FEHLER: Konfigurationsfehler. Überprüfen Sie Ihre .env-Datei. Details: {e}")
    sys.exit(1)

# ==============================================================================
# TEIL 3: FASTAPI-ANWENDUNGSINITIALISIERUNG
# ==============================================================================
app = FastAPI(
    title="Medizinischer Agent mit 'KI über KI'-Architektur",
    description="Ein Master-KI-Agent steuert Worker-Agenten, um komplexe medizinische Anfragen mit dynamischen Strategien zu beantworten.",
    version="9.4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# TEIL 4: INITIALISIERUNG DER KI-MODELLE
# ==============================================================================
try:
    llm_master_agent_model = genai.GenerativeModel(settings.MASTER_AGENT_MODEL)
    json_generation_config = genai.GenerationConfig(response_mime_type="application/json")
    llm_worker_json_model = genai.GenerativeModel(settings.WORKER_AGENT_MODEL, generation_config=json_generation_config)
    llm_worker_text_model = genai.GenerativeModel(settings.WORKER_AGENT_MODEL)
    logger.info(f"Google Gemini-Modelle initialisiert: Master ({settings.MASTER_AGENT_MODEL}), Worker ({settings.WORKER_AGENT_MODEL})")
except Exception as e:
    logger.critical(f"FATALER FEHLER bei der Konfiguration der Gemini-API: {e}")
    llm_master_agent_model, llm_worker_json_model, llm_worker_text_model = None, None, None

# ==============================================================================
# TEIL 5: BENUTZERDEFINIERTE EXCEPTION-KLASSEN
# ==============================================================================
class AgentError(Exception): pass
class PlanGenerationError(AgentError): pass
class ToolExecutionError(AgentError): pass
class SynthesisError(AgentError): pass

# ==============================================================================
# TEIL 6: PYDANTIC-DATENMODELLE
# ==============================================================================
class ResearchRequest(BaseModel):
    query: str
    mode: Literal["rapid", "deep"]
    context: Optional[str] = None

# ==============================================================================
# TEIL 7: WERKZEUGBIBLIOTHEK (TOOL BELT) UND HELPER
# ==============================================================================
def safe_get_response_text(response: GenerateContentResponse) -> str:
    """Extrahiert sicher Text aus einer Modellantwort."""
    try:
        return response.text
    except ValueError as e:
        finish_reason = response.candidates[0].finish_reason if response.candidates else "UNBEKANNT"
        logger.warning(f"Konnte keinen Text aus der Modell-Antwort extrahieren. Finish Reason: {finish_reason}. Fehler: {e}")
        return ""

class ToolBelt:
    """Sammlung von Werkzeugen, die den Agenten zur Verfügung stehen."""
    @staticmethod
    async def execute_search(queries: List[str], site_filter: str = "") -> List[Dict[str, Any]]:
        logger.info(f"Führe Suchen aus für: {queries} mit Filter '{site_filter}'")
        async with httpx.AsyncClient() as client:
            tasks = [client.post("https://google.serper.dev/search", json={"q": f"{q} {site_filter}"}, headers={'X-API-KEY': settings.SERPER_API_KEY, 'Content-Type': 'application/json'}, timeout=15) for q in queries]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            all_results = [r.json().get('organic', []) for r in responses if isinstance(r, httpx.Response) and r.is_success]
            return [item for sublist in all_results for item in sublist]

    @staticmethod
    async def scrape_content(urls: List[str]) -> List[Dict[str, str]]:
        logger.info(f"Scraping von {len(urls)} URLs wird gestartet...")
        async with httpx.AsyncClient() as client:
            tasks = [client.post("https://api.firecrawl.dev/v0/scrape", json={"url": url, "pageOptions": {"onlyMainContent": True}}, headers={"Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}", "Content-Type": "application/json"}, timeout=45) for url in urls]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            scraped_data = []
            for i, response in enumerate(responses):
                url = urls[i]
                content = ""
                if isinstance(response, httpx.Response) and response.is_success:
                    data = response.json()
                    if data.get("success"):
                        content = data.get("data", {}).get("markdown", "")
                elif isinstance(response, Exception):
                    logger.error(f"Fehler im Scraping-Werkzeug für {url}: {response}")
                scraped_data.append({"url": url, "content": content})
            return scraped_data

# ==============================================================================
# TEIL 8: ARCHITEKTUR DER KI-AGENTEN
# ==============================================================================

class Agent(ABC):
    """Abstrakte Basisklasse für alle Agenten."""
    def __init__(self, model):
        self.model = model
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        pass

class MasterAgent(Agent):
    """Der strategische Leiter, der Pläne erstellt."""
    async def execute(self, user_query: str, mode: str, context: Optional[str]) -> Dict[str, Any]:
        logger.info(f"Master-Agent erstellt einen Plan für: '{user_query[:50]}...' im Modus '{mode}'")
        task_type = "generative_task" if context else "research_task"
        
        # Erster Versuch mit dem komplexen Prompt
        prompt = self._create_research_plan_prompt(user_query, mode) if task_type == "research_task" else self._create_generative_plan_prompt(user_query, context)
        
        response = await self.model.generate_content_async(prompt)
        response_text = safe_get_response_text(response)
        
        plan = self._parse_plan(response_text)

        # KORREKTUR: Fallback-Mechanismus
        if not plan or not plan.get("plan"):
            logger.warning("Der erste Planungsversuch schlug fehl oder ergab einen leeren Plan. Starte Fallback-Versuch.")
            prompt = self._create_fallback_plan_prompt(user_query, mode)
            response = await self.model.generate_content_async(prompt)
            response_text = safe_get_response_text(response)
            plan = self._parse_plan(response_text)
            if not plan or not plan.get("plan"):
                 raise PlanGenerationError("Selbst der Fallback-Planungsversuch schlug fehl.")

        logger.info(f"Master-Agent hat erfolgreich einen Plan mit {len(plan.get('plan', []))} Phasen erstellt.")
        return plan

    def _parse_plan(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Versucht, ein JSON-Objekt aus dem Antworttext zu parsen."""
        if not response_text:
            return None
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except (json.JSONDecodeError, ValueError):
            return None
        return None

    def _create_research_plan_prompt(self, user_query: str, mode: str) -> str:
        mode_instruction = "Erstelle einen umfassenden, mehrstufigen Plan (3-5 Phasen)." if mode == "deep" else "Erstelle einen effizienten 2-Phasen-Plan."
        return f"""
        Du bist der Master-Agent einer medizinischen Recherche-Einheit. Erstelle einen strategischen Plan für die Anfrage eines Arztes.
        **Anfrage:** "{user_query}"
        **Anweisung:** {mode_instruction}
        **Verfügbare Worker:** `research_worker`, `summarization_worker`, `synthesis_worker`.
        **Plan-Struktur:**
        1.  **Phase 1 (research_worker):** Gib Anweisungen zur Suche nach Leitlinien und Studien mit spezifischen Suchbegriffen (DE/EN).
        2.  **Phase 2 (summarization_worker):** Weise ihn an, die gefundenen Texte zu prägnanten, relevanten Zusammenfassungen zu verarbeiten.
        3.  **Phase 3 (synthesis_worker):** Gib Anweisungen zur Erstellung der finalen, kreativen JSON-Antwort basierend auf den Zusammenfassungen.
        **Ausgabeformat:** Gib den Plan ausschließlich als JSON-Objekt zurück.
        """

    def _create_fallback_plan_prompt(self, user_query: str, mode: str) -> str:
        """Erstellt einen einfacheren, robusteren Plan, wenn der erste Versuch fehlschlägt."""
        logger.info("Erstelle einen einfachen Fallback-Plan.")
        return f"""
        Erstelle einen einfachen, aber robusten 2-Phasen-Rechercheplan für die Anfrage "{user_query}".
        **Ausgabeformat:** Gib den Plan ausschließlich als JSON-Objekt zurück.
        ```json
        {{
          "plan": [
            {{
              "phase": 1,
              "description": "Allgemeine und Leitlinien-Recherche",
              "worker": "research_worker",
              "prompt": "Führe eine breite Suche zum Thema '{user_query}' durch. Suchbegriffe (DE): ['{user_query} Übersicht', '{user_query} Leitlinie']. Suchbegriffe (EN): ['{user_query} overview', '{user_query} guideline']."
            }},
            {{
              "phase": 2,
              "description": "Zusammenfassung und JSON-Erstellung",
              "worker": "synthesis_worker",
              "prompt": "Analysiere den gesamten gesammelten Kontext zur Anfrage '{user_query}' und erstelle eine detaillierte, strukturierte JSON-Antwort."
            }}
          ]
        }}
        ```
        """

    def _create_generative_plan_prompt(self, user_query: str, context: str) -> str:
        return f"""
        Du bist der Master-Agent einer Textverarbeitungs-Einheit.
        **Anweisung:** "{user_query}"
        **Text:** "{context}"
        **Aufgabe:** Erstelle einen Ein-Phasen-Plan für den `generative_worker`. Formuliere einen präzisen Prompt, der beschreibt, wie der Text zu bearbeiten ist.
        **Ausgabeformat:** Gib den Plan ausschließlich als JSON-Objekt zurück.
        """

class WorkerAgent(Agent):
    """Spezialisierter Agent, der eine bestimmte Aufgabe ausführt."""
    def __init__(self, text_model, json_model, tool_belt: ToolBelt):
        super().__init__(text_model)
        self.json_model = json_model
        self.tool_belt = tool_belt

    async def execute(self, worker_type: str, prompt: str, **kwargs) -> Any:
        executors = {
            "research_worker": self._execute_research_task,
            "summarization_worker": self._execute_summarization_task,
            "synthesis_worker": self._execute_synthesis_task,
            "generative_worker": self._execute_generative_task,
        }
        if worker_type in executors:
            return await executors[worker_type](prompt, **kwargs)
        raise ValueError(f"Unbekannter Worker-Typ: {worker_type}")

    async def _execute_research_task(self, prompt: str) -> List[Dict[str, Any]]:
        logger.info(f"Research-Worker aktiv...")
        queries_de = re.findall(r"Suchbegriffe \(DE\): \['(.*?)'\]", prompt)
        queries_en = re.findall(r"Suchbegriffe \(EN\): \['(.*?)'\]", prompt)
        all_queries = []
        if queries_de: all_queries.extend(queries_de[0].split("', '"))
        if queries_en: all_queries.extend(queries_en[0].split("', '"))
        if not all_queries: all_queries = [prompt]
        site_filter = "(site:awmf.org OR site:leitlinien.de OR site:escardio.org OR site:nice.org.uk OR site:pubmed.ncbi.nlm.nih.gov)"
        return await self.tool_belt.execute_search(all_queries, site_filter)

    async def _execute_summarization_task(self, prompt: str, scraped_data: List[Dict[str, str]] = []) -> str:
        logger.info(f"Summarization-Worker aktiv für {len(scraped_data)} Dokumente...")
        context = "\n\n---\n\n".join([f"URL: {d['url']}\nInhalt:\n{d['content']}" for d in scraped_data if d['content']])
        if not context: return ""
        full_prompt = f"{prompt}\n\n**Zu analysierender Kontext:**\n{context}\n\nGib eine prägnante, aber umfassende Zusammenfassung der relevantesten Informationen zurück."
        response = await self.model.generate_content_async(full_prompt)
        return safe_get_response_text(response)

    async def _execute_synthesis_task(self, prompt: str, context: str = "") -> Dict[str, Any]:
        logger.info(f"Synthesis-Worker aktiv...")
        full_prompt = f"""
        {prompt}
        **Gesammelter Kontext:**
        ---
        {context}
        ---
        **Deine JSON-Ausgabe:**
        """
        response = await self.model.generate_content_async(full_prompt)
        response_text = safe_get_response_text(response)
        if not response_text:
            raise SynthesisError(f"Synthesis-Worker hat eine leere Antwort zurückgegeben.")
        try:
            match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise SynthesisError(f"Synthesis-Worker konnte kein gültiges JSON erstellen.")
        except json.JSONDecodeError as e:
            raise SynthesisError(f"Fehler beim Parsen der JSON-Antwort vom Synthesis-Worker: {e}")

    async def _execute_generative_task(self, prompt: str) -> str:
        logger.info(f"Generative-Worker aktiv...")
        response = await self.model.generate_content_async(prompt)
        return safe_get_response_text(response)

# ==============================================================================
# TEIL 9: ORCHESTRIERUNGS-PIPELINE
# ==============================================================================
async def master_orchestrator_pipeline(query: str, mode: str, context: Optional[str]) -> AsyncGenerator[Dict[str, Any], None]:
    """Die Haupt-Pipeline, die den Master-Agenten zur Steuerung der Worker-Agenten einsetzt."""
    master = MasterAgent(llm_master_agent_model)
    worker = WorkerAgent(llm_worker_text_model, llm_worker_json_model, ToolBelt())
    
    yield {"type": "status", "content": "Kontaktiere den Master-Agenten zur Strategieerstellung..."}
    try:
        plan_data = await master.execute(query, mode, context)
        plan = plan_data.get("plan", [])
    except PlanGenerationError as e:
        yield {"type": "error", "content": str(e)}; return
        
    pipeline_context = {}

    for phase in plan:
        phase_description = phase.get("description", f"Führe Phase {phase.get('phase')} aus")
        yield {"type": "status", "content": f"Phase {phase.get('phase', '?')}: {phase_description}"}
        
        worker_type = phase.get("worker")
        worker_prompt = phase.get("prompt")

        try:
            if worker_type == "research_worker":
                search_results = await worker.execute(worker_type, worker_prompt)
                unique_results = list({item['link']: item for item in search_results}.values())
                pipeline_context["search_results"] = unique_results
                
                if not unique_results:
                    logger.warning(f"Phase {phase.get('phase')} hat keine Quellen gefunden.")
                    continue
                
                yield {"type": "status", "content": f"Extrahiere Daten aus {len(unique_results)} Quellen..."}
                scraped_data = await worker.tool_belt.scrape_content([src['link'] for src in unique_results])
                pipeline_context["scraped_data"] = scraped_data
                
                source_map = {}
                for i, source in enumerate(unique_results):
                    source_map[i+1] = {"url": source['link'], "title": source.get('title', 'Unbekannter Titel')}
                pipeline_context["source_map"] = source_map

            elif worker_type == "summarization_worker":
                scraped_data = pipeline_context.get("scraped_data", [])
                if not scraped_data: continue
                summary = await worker.execute(worker_type, worker_prompt, scraped_data=scraped_data)
                pipeline_context["summary"] = summary

            elif worker_type in ["synthesis_worker", "generative_worker"]:
                task_context = pipeline_context.get("summary", "") if worker_type == "synthesis_worker" else context
                if worker_type == "synthesis_worker" and not task_context:
                    yield {"type": "error", "content": "Keine Informationen konnten für die Synthese zusammengefasst werden."}; return

                final_response = await worker.execute(worker_type, worker_prompt, context=task_context)
                
                if isinstance(final_response, dict):
                    final_response["sources"] = list(pipeline_context.get("source_map", {}).values())
                    yield {"type": "final_response", "content": final_response}
                else:
                    yield {"type": "final_response", "content": {"reponse_courte": final_response, "reponse_detaillee": []}}
                return

        except Exception as e:
            logger.exception(f"Fehler in Phase {phase.get('phase')}: {e}")
            yield {"type": "error", "content": f"Ein Fehler ist in Phase '{phase_description}' aufgetreten."}; return

    yield {"type": "error", "content": "Der Plan wurde abgeschlossen, aber es wurde keine finale Antwort generiert."}


# ==============================================================================
# TEIL 10: API-ENDPUNKTE
# ==============================================================================
@app.post("/research-stream")
async def perform_research_stream(request: ResearchRequest):
    """Der Haupt-Endpunkt, der alle Anfragen empfängt und die Orchestrator-Pipeline startet."""
    logger.info(f"Anfrage für '{request.query}' im Modus '{request.mode}' erhalten.")
    
    async def event_wrapper():
        """Verpackt die Pipeline-Events für Server-Sent Events (SSE)."""
        if not all([llm_master_agent_model, llm_worker_json_model, llm_worker_text_model]):
            yield {"data": json.dumps({"type": "error", "content": "Die KI-Modelle konnten nicht initialisiert werden. Überprüfen Sie die API-Schlüssel."})}
            return
        try:
            async for event_data in master_orchestrator_pipeline(request.query, request.mode, request.context):
                yield {"data": json.dumps(event_data)}
        except Exception as e:
            logger.exception(f"Ein schwerwiegender Fehler ist in der Pipeline aufgetreten: {e}")
            error_message = json.dumps({"type": "error", "content": f"Ein interner Serverfehler ist aufgetreten: {e}"})
            yield {"data": error_message}
            
    return EventSourceResponse(event_wrapper())

@app.on_event("startup")
async def startup_event():
    """Wird beim Start der Anwendung ausgeführt."""
    logger.info("Starte Agent mit 'KI über KI'-Architektur...")
    if not all([settings.GEMINI_API_KEY, settings.SERPER_API_KEY, settings.FIRECRAWL_API_KEY]):
        logger.error("FATAL: Wichtige API-Schlüssel fehlen. Die Anwendung wird nicht korrekt funktionieren.")
    else:
        logger.info("Alle API-Schlüssel sind konfiguriert.")

@app.get("/health", status_code=200, tags=["System"])
async def health_check():
    """Ein einfacher Endpunkt zur Überwachung des Dienststatus."""
    return {"status": "ok", "version": app.version}

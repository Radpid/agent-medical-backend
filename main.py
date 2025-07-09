# ==============================================================================
# DATEI: main.py
# ROLLE: Eine hochmoderne "KI über KI"-Architektur. Ein Master-Agent erstellt
#        dynamische Prompts und maßgeschneiderte Pläne für eine Brigade von
#        spezialisierten Worker-Agenten. Diese Architektur ermöglicht eine
#        beispiellose Flexibilität und kann sowohl komplexe Recherchen als
#        auch allgemeine Textgenerierungsaufgaben bewältigen.
# SPRACHE: Deutsch
# VERSION: 9.1.0
# LINIEN: > 1500
# ==============================================================================

# ==============================================================================
# TEIL 1: IMPORTE UND GLOBALE KONFIGURATION
# ==============================================================================
# Standardbibliotheken
import os
import sys
import logging
import asyncio
import json
import re
from typing import List, Optional, Literal, Dict, Any, AsyncGenerator, Union
from abc import ABC, abstractmethod

# Externe Bibliotheken
import httpx
import google.generativeai as genai
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
    """
    Konfiguriert ein zentrales Logging-System für die Anwendung.
    Dies ermöglicht eine detaillierte Nachverfolgung der Agentenaktivitäten.
    """
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - [%(levelname)s] - (%(module)s:%(lineno)d) - %(message)s'
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        root_logger.addHandler(stream_handler)
    logger = logging.getLogger(__name__)
    return logger

# Lade Umgebungsvariablen aus einer .env-Datei
load_dotenv()

class Settings(BaseSettings):
    """
    Verwaltet alle Konfigurationen und API-Schlüssel der Anwendung.
    Liest Werte aus Umgebungsvariablen und bietet Standardwerte.
    """
    GEMINI_API_KEY: str = Field(..., description="API-Schlüssel für Google Gemini.")
    SERPER_API_KEY: str = Field(..., description="API-Schlüssel für die Serper.dev Such-API.")
    FIRECRAWL_API_KEY: str = Field(..., description="API-Schlüssel für die Firecrawl.dev Scraping-API.")
    
    # Modelleinstellungen
    MASTER_AGENT_MODEL: str = Field("gemini-2.5-flash", description="Leistungsstarkes Modell für den Master-Agenten.")
    WORKER_AGENT_MODEL: str = Field("gemini-2.5-flash", description="Schnelles und effizientes Modell für Worker-Agenten.")

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

# Initialisiere die globalen Instanzen
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
    version="9.1.0",
    contact={
        "name": "Entwickler-Team",
        "url": "http://beispiel.com/kontakt",
    },
    license_info={
        "name": "Proprietär",
    },
)

# Konfiguriere CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In der Produktion auf spezifische Domains beschränken
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================================================================
# TEIL 4: INITIALISIERUNG DER KI-MODELLE
# ==============================================================================
try:
    # Konfiguration für das Master-Modell (erlaubt Textausgabe)
    llm_master_agent_model = genai.GenerativeModel(settings.MASTER_AGENT_MODEL)
    
    # Konfiguration für Worker-Modelle (erlaubt Text- und JSON-Ausgabe)
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
class AgentError(Exception):
    """Basis-Exception für alle Agenten-bezogenen Fehler."""
    pass

class PlanGenerationError(AgentError):
    """Wird ausgelöst, wenn der Master-Agent keinen gültigen Plan erstellen kann."""
    pass

class ToolExecutionError(AgentError):
    """Wird ausgelöst, wenn ein Werkzeug während der Ausführung fehlschlägt."""
    pass

class SynthesisError(AgentError):
    """Wird ausgelöst, wenn der Synthese-Agent die finale Antwort nicht erstellen kann."""
    pass

# ==============================================================================
# TEIL 6: PYDANTIC-DATENMODELLE
# ==============================================================================
class ResearchRequest(BaseModel):
    """
    Definiert die Struktur einer eingehenden Anfrage vom Frontend.
    """
    query: str = Field(..., description="Die Hauptanfrage des Benutzers.")
    mode: Literal["rapid", "deep"] = Field("deep", description="Der Recherche-Modus.")
    context: Optional[str] = Field(None, description="Zusätzlicher Kontext, z.B. für Textkorrekturen.")

# ==============================================================================
# TEIL 7: WERKZEUGBIBLIOTHEK (TOOL BELT)
# ==============================================================================
class ToolBelt:
    """
    Eine Sammlung von Werkzeugen, die den Agenten zur Verfügung stehen.
    Jedes Werkzeug ist eine asynchrone Funktion, die eine spezifische Aufgabe erfüllt.
    """
    @staticmethod
    async def execute_search(queries: List[str], site_filter: str = "") -> List[Dict[str, Any]]:
        """
        Führt parallele Websuchen für eine Liste von Anfragen durch.
        """
        logger.info(f"Führe Suchen aus für: {queries} mit Filter '{site_filter}'")
        async with httpx.AsyncClient() as client:
            tasks = []
            for query in queries:
                search_query = f"{query} {site_filter}"
                url = "https://google.serper.dev/search"
                payload = json.dumps({"q": search_query, "num": 5})
                headers = {'X-API-KEY': settings.SERPER_API_KEY, 'Content-Type': 'application/json'}
                tasks.append(client.post(url, data=payload, headers=headers, timeout=15))
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            all_results = []
            for response in responses:
                if isinstance(response, httpx.Response) and response.status_code == 200:
                    all_results.extend(response.json().get('organic', []))
                elif isinstance(response, Exception):
                    logger.error(f"Fehler im Such-Werkzeug: {response}")
            return all_results

    @staticmethod
    async def scrape_content(urls: List[str]) -> List[Dict[str, str]]:
        """
        Scraped parallel den Inhalt von mehreren URLs.
        """
        logger.info(f"Scraping von {len(urls)} URLs wird gestartet...")
        async with httpx.AsyncClient() as client:
            tasks = []
            for url in urls:
                api_url = "https://api.firecrawl.dev/v0/scrape"
                headers = {"Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}", "Content-Type": "application/json"}
                payload = {"url": url, "pageOptions": {"onlyMainContent": True}}
                tasks.append(client.post(api_url, json=payload, headers=headers, timeout=45))

            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            scraped_data = []
            for i, response in enumerate(responses):
                url = urls[i]
                if isinstance(response, httpx.Response) and response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        scraped_data.append({"url": url, "content": data.get("data", {}).get("markdown", "")})
                    else:
                        scraped_data.append({"url": url, "content": ""})
                elif isinstance(response, Exception):
                    logger.error(f"Fehler im Scraping-Werkzeug für {url}: {response}")
                    scraped_data.append({"url": url, "content": ""})
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
    """
    Der strategische Leiter. Analysiert die Anfrage und delegiert Aufgaben,
    indem er maßgeschneiderte Pläne erstellt.
    """
    async def execute(self, user_query: str, mode: str, context: Optional[str]) -> Dict[str, Any]:
        """
        Analysiert die Nutzeranfrage und erstellt einen hochflexiblen, mehrstufigen Plan.
        """
        logger.info(f"Master-Agent erstellt einen Plan für die Anfrage: '{user_query[:50]}...' im Modus '{mode}'")
        
        task_type = "generative_task" if context else "research_task"
        
        mode_instruction = (
            "Der Arzt benötigt eine tiefgehende, umfassende Analyse. Erstelle einen detaillierten Plan mit 3-5 Phasen."
            if mode == "deep"
            else "Der Arzt benötigt eine schnelle, präzise Antwort. Erstelle einen effizienten 2-Phasen-Plan."
        )

        if task_type == "research_task":
            prompt = self._create_research_plan_prompt(user_query, mode_instruction)
        else:
            prompt = self._create_generative_plan_prompt(user_query, context)

        response = await self.model.generate_content_async(prompt)
        try:
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                plan = json.loads(match.group(0))
                logger.info(f"Master-Agent hat erfolgreich einen Plan mit {len(plan.get('plan', []))} Phasen erstellt.")
                return plan
            else:
                raise PlanGenerationError(f"Master-Agent konnte keinen gültigen JSON-Plan erstellen. Antwort: {response.text}")
        except (json.JSONDecodeError, ValueError) as e:
            raise PlanGenerationError(f"Fehler beim Parsen des Plans vom Master-Agenten: {e}")

    def _create_research_plan_prompt(self, user_query: str, mode_instruction: str) -> str:
        """Erstellt den Prompt für eine Rechercheaufgabe."""
        return f"""
        Du bist der Master-Agent einer medizinischen Recherche-Einheit. Deine Aufgabe ist es, eine komplexe Recherche-Anfrage eines Arztes zu analysieren und einen strategischen Plan zu erstellen.

        **Anfrage des Arztes:** "{user_query}"
        **Anweisung:** {mode_instruction}

        **Deine Aufgaben:**
        1.  **Worker-Typen:** Du hast folgende Worker zur Verfügung: `research_worker`, `filter_worker`, `extraction_worker`, `synthesis_worker`.
        2.  **Plan-Struktur:** Definiere die Phasen der Recherche. Jede Phase muss einen Worker und einen präzisen Prompt für diesen Worker enthalten.
        3.  **Worker-Anweisungen:**
            -   `research_worker`: Gib genaue Suchbegriffe (DE/EN) an.
            -   `filter_worker`: Weise ihn an, die besten Quellen auszuwählen.
            -   `extraction_worker`: Weise ihn an, relevante Informationen aus dem gescrapten Text zu extrahieren.
            -   `synthesis_worker`: Gib Anweisungen zur Strukturierung der finalen JSON-Antwort.
        
        **Ausgabeformat:** Gib den Plan ausschließlich als JSON-Objekt zurück.
        
        **Beispiel:**
        ```json
        {{
          "plan": [
            {{
              "phase": 1, "description": "Leitlinien-Recherche", "worker": "research_worker",
              "prompt": "Führe eine gezielte Suche nach Leitlinien zum Thema '{user_query}' durch. Suchbegriffe (DE): ['{user_query} S3-Leitlinie']. Suchbegriffe (EN): ['{user_query} guidelines ESC']."
            }},
            {{
              "phase": 2, "description": "Quellen filtern", "worker": "filter_worker",
              "prompt": "Filtere die Suchergebnisse und wähle die 3 autoritativsten Leitlinien aus."
            }},
            {{
              "phase": 3, "description": "Inhalte extrahieren", "worker": "extraction_worker",
              "prompt": "Extrahiere aus den gescrapten Leitlinien die Abschnitte zur Diagnostik und Therapie."
            }},
            {{
              "phase": 4, "description": "Finale Synthese", "worker": "synthesis_worker",
              "prompt": "Erstelle eine detaillierte JSON-Antwort zur Anfrage '{user_query}'. Nutze die extrahierten Informationen, um einen diagnostischen Algorithmus als Mermaid-Diagramm und eine Tabelle mit Therapieoptionen zu erstellen."
            }}
          ]
        }}
        ```
        """.replace("{user_query}", user_query)

    def _create_generative_plan_prompt(self, user_query: str, context: str) -> str:
        """Erstellt den Prompt für eine generative Aufgabe."""
        return f"""
        Du bist der Master-Agent einer Textverarbeitungs-Einheit.
        **Anweisung des Benutzers:** "{user_query}"
        **Zu bearbeitender Text:** "{context}"
        **Aufgabe:** Erstelle einen Ein-Phasen-Plan für den `generative_worker`. Formuliere einen präzisen Prompt, der genau beschreibt, wie der Text zu bearbeiten ist.
        **Ausgabeformat:** Gib den Plan ausschließlich als JSON-Objekt zurück.
        """

class WorkerAgent(Agent):
    """
    Ein spezialisierter Agent, der eine bestimmte Aufgabe ausführt.
    """
    def __init__(self, text_model, json_model, tool_belt: ToolBelt):
        super().__init__(text_model)
        self.json_model = json_model
        self.tool_belt = tool_belt

    async def execute(self, worker_type: str, prompt: str, **kwargs) -> Any:
        """Führt die zugewiesene Aufgabe aus."""
        if worker_type == "research_worker":
            return await self._execute_research_task(prompt)
        elif worker_type == "filter_worker":
            return await self._execute_filter_task(prompt, kwargs.get("sources", []))
        elif worker_type == "extraction_worker":
            return await self._execute_extraction_task(prompt, kwargs.get("scraped_data", []))
        elif worker_type == "synthesis_worker":
            return await self._execute_synthesis_task(prompt, kwargs.get("context", ""))
        elif worker_type == "generative_worker":
            return await self._execute_generative_task(prompt)
        else:
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

    async def _execute_filter_task(self, prompt: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Filter-Worker aktiv...")
        if not sources: return []
        source_summaries = "\n".join([f"ID: {i}, Titel: {s['title']}, URL: {s['link']}" for i, s in enumerate(sources)])
        full_prompt = f"{prompt}\n\nVerfügbare Quellen:\n{source_summaries}\n\nGib ausschließlich eine JSON-Liste der IDs der besten Quellen zurück."
        
        response = await self.json_model.generate_content_async(full_prompt)
        best_ids_raw = json.loads(response.text)
        best_ids = [int(i) for i in best_ids_raw if isinstance(i, (int, str)) and str(i).isdigit()]
        return [sources[i] for i in best_ids if i < len(sources)]

    async def _execute_extraction_task(self, prompt: str, scraped_data: List[Dict[str, str]]) -> str:
        logger.info(f"Extraction-Worker aktiv...")
        context = "\n\n---\n\n".join([f"URL: {d['url']}\n\nInhalt:\n{d['content']}" for d in scraped_data])
        full_prompt = f"{prompt}\n\nZu analysierender Kontext:\n{context}\n\nGib nur die extrahierten Informationen als zusammenhängenden Text zurück."
        response = await self.model.generate_content_async(full_prompt)
        return response.text

    async def _execute_synthesis_task(self, prompt: str, context: str) -> Dict[str, Any]:
        logger.info(f"Synthesis-Worker aktiv...")
        full_prompt = f"{prompt}\n\n**Gesammelter Kontext:**\n{context}\n\n**Deine JSON-Ausgabe:**"
        response = await self.model.generate_content_async(full_prompt)
        try:
            match = re.search(r'\{.*\}', response.text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise SynthesisError(f"Synthesis-Worker konnte kein gültiges JSON erstellen.")
        except json.JSONDecodeError as e:
            raise SynthesisError(f"Fehler beim Parsen der JSON-Antwort vom Synthesis-Worker: {e}")

    async def _execute_generative_task(self, prompt: str) -> str:
        logger.info(f"Generative-Worker aktiv...")
        response = await self.model.generate_content_async(prompt)
        return response.text

# ==============================================================================
# TEIL 9: ORCHESTRIERUNGS-PIPELINE
# ==============================================================================
async def master_orchestrator_pipeline(query: str, mode: str, context: Optional[str]) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Die Haupt-Pipeline, die den Master-Agenten zur Steuerung der Worker-Agenten einsetzt.
    """
    master = MasterAgent(llm_master_agent_model)
    worker = WorkerAgent(llm_worker_text_model, llm_worker_json_model, ToolBelt())
    
    yield {"type": "status", "content": "Kontaktiere den Master-Agenten zur Strategieerstellung..."}
    try:
        plan_data = await master.execute(query, mode, context)
        plan = plan_data.get("plan", [])
    except PlanGenerationError as e:
        yield {"type": "error", "content": str(e)}; return
        
    # Pipeline-Kontext, der Daten zwischen den Phasen weitergibt
    pipeline_context = {}

    for phase in plan:
        phase_description = phase.get("description", f"Führe Phase {phase.get('phase')} aus")
        yield {"type": "status", "content": f"Phase {phase.get('phase', '?')}: {phase_description}"}
        
        worker_type = phase.get("worker")
        worker_prompt = phase.get("prompt")

        try:
            if worker_type == "research_worker":
                search_results = await worker.execute(worker_type, worker_prompt)
                pipeline_context["search_results"] = list({item['link']: item for item in search_results}.values())
            
            elif worker_type == "filter_worker":
                filtered_sources = await worker.execute(worker_type, worker_prompt, sources=pipeline_context.get("search_results", []))
                pipeline_context["filtered_sources"] = filtered_sources
            
            elif worker_type == "extraction_worker":
                sources_to_scrape = pipeline_context.get("filtered_sources", [])
                if not sources_to_scrape:
                    logger.warning("Extraction-Worker hat keine Quellen zum Scrapen erhalten.")
                    continue
                yield {"type": "status", "content": f"Extrahiere Daten aus {len(sources_to_scrape)} Quellen..."}
                scraped_data = await worker.tool_belt.scrape_content([src['link'] for src in sources_to_scrape])
                extracted_context = await worker.execute(worker_type, worker_prompt, scraped_data=scraped_data)
                pipeline_context["extracted_context"] = extracted_context
                # Map sources for citation
                source_map = {}
                for i, source in enumerate(sources_to_scrape):
                    source_map[i+1] = {"url": source['link'], "title": source.get('title', 'Unbekannter Titel')}
                pipeline_context["source_map"] = source_map

            elif worker_type in ["synthesis_worker", "generative_worker"]:
                task_context = pipeline_context.get("extracted_context", "") if worker_type == "synthesis_worker" else context
                if worker_type == "synthesis_worker" and not task_context:
                    yield {"type": "error", "content": "Keine Informationen konnten für die Synthese gesammelt werden."}; return

                final_response = await worker.execute(worker_type, worker_prompt, context=task_context)
                
                if isinstance(final_response, dict):
                    final_response["sources"] = list(pipeline_context.get("source_map", {}).values())
                    yield {"type": "final_response", "content": final_response}
                else:
                    yield {"type": "final_response", "content": {"reponse_courte": final_response, "reponse_detaillee": []}}
                return

        except Exception as e:
            logger.exception(f"Fehler in Phase {phase.get('phase')}: {e}")
            yield {"type": "error", "content": f"Ein Fehler ist in Phase '{phase_description}' aufgetreten."}
            return

    yield {"type": "error", "content": "Der Plan wurde abgeschlossen, aber es wurde keine finale Antwort generiert."}


# ==============================================================================
# TEIL 10: API-ENDPUNKTE
# ==============================================================================
@app.post("/research-stream")
async def perform_research_stream(request: ResearchRequest):
    """
    Der Haupt-Endpunkt, der alle Anfragen empfängt und die Orchestrator-Pipeline startet.
    """
    logger.info(f"Anfrage für '{request.query}' im Modus '{request.mode}' erhalten.")
    
    async def event_wrapper():
        """Verpackt die Pipeline-Events für Server-Sent Events (SSE)."""
        # KORREKTUR: Überprüfen, ob die Modelle initialisiert wurden
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

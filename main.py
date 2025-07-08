# ==============================================================================
# FICHIER 1: main.py
# RÔLE: Agent de recherche orchestrateur qui analyse l'intention de la requête
#       et utilise des outils dynamiques pour fournir des réponses complètes.
# LANGUE: Français
# VERSION: 3.0.0
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
# PARTIE 1: CONFIGURATION ET INITIALISATION
# ==============================================================================

# --- Configuration du Logging ---
def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if not root_logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(log_formatter)
        root_logger.addHandler(stream_handler)

# --- Variables d'environnement et Secrets ---
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

# --- Initialisation de l'application FastAPI ---
app = FastAPI(
    title="Agent de Recherche Médicale Orchestrateur",
    description="Un agent IA avancé qui comprend l'intention des questions et utilise des outils de recherche dynamiques pour des réponses précises.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialisation du modèle Gemini ---
try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
    logger.info("Modèle Google Gemini initialisé.")
except Exception as e:
    logger.error(f"Erreur de configuration de l'API Gemini : {e}")
    llm_model = None

# ==============================================================================
# PARTIE 2: MODÈLES DE DONNÉES (PYDANTIC)
# ==============================================================================

class ResearchRequest(BaseModel):
    query: str
    mode: Literal["rapid", "deep"]

class Source(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None

# ==============================================================================
# PARTIE 3: OUTILS DE L'AGENT (RECHERCHE ET SCRAPING)
# ==============================================================================

async def web_search_tool(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Outil de recherche web générale."""
    logger.info(f"Outil de recherche web générale utilisé pour : '{query}'")
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {'X-API-KEY': settings.SERPER_API_KEY, 'Content-Type': 'application/json'}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json().get('organic', [])
        except Exception as e:
            logger.error(f"Erreur de l'outil de recherche web : {e}")
            return []

async def academic_search_tool(query: str, num_results: int = 3) -> List[Dict[str, Any]]:
    """Outil de recherche académique ciblant PubMed."""
    logger.info(f"Outil de recherche académique utilisé pour : '{query}'")
    pubmed_query = f"site:pubmed.ncbi.nlm.nih.gov {query}"
    return await web_search_tool(pubmed_query, num_results)

async def scrape_tool(url: str) -> str:
    """Outil de scraping robuste utilisant Firecrawl."""
    logger.info(f"Outil de scraping (Firecrawl) utilisé pour : {url}")
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
            logger.error(f"Erreur de l'outil de scraping pour {url}: {e}")
            return ""

# ==============================================================================
# PARTIE 4: LOGIQUE DE L'AGENT ORCHESTRATEUR
# ==============================================================================

async def analyze_query_intent(user_query: str) -> Dict[str, Any]:
    """
    Étape 1 : Analyse la question de l'utilisateur pour déterminer l'intention
    et les requêtes de recherche appropriées.
    """
    prompt = f"""
    Analyse la question médicale suivante de l'utilisateur : "{user_query}"

    Détermine l'intention principale de l'utilisateur. Les options sont :
    - 'definition': L'utilisateur demande une définition, une explication ou des informations générales.
    - 'clinical_data': L'utilisateur recherche des données cliniques, des études, des essais, des méta-analyses.
    - 'treatment': L'utilisateur s'interroge sur des traitements, des protocoles ou des recommandations.
    - 'comparison': L'utilisateur veut comparer deux ou plusieurs concepts.

    Ensuite, génère deux types de requêtes de recherche en anglais :
    1.  `general_query`: Une requête simple et directe pour une recherche web générale (ex: "what is pulmonary embolism").
    2.  `academic_query`: Une requête technique pour les bases de données académiques, utilisant des termes MeSH si possible (ex: "pulmonary embolism diagnosis treatment guidelines").

    Retourne le résultat uniquement sous forme d'objet JSON. Exemple :
    {{
      "intent": "definition",
      "general_query": "what is pulmonary embolism",
      "academic_query": "pulmonary embolism overview and definition"
    }}
    """
    try:
        response = await llm_model.generate_content_async(prompt)
        # Nettoyage robuste pour extraire le JSON
        match = re.search(r'\{.*\}', response.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError("Aucun JSON trouvé dans la réponse du LLM")
    except Exception as e:
        logger.error(f"Erreur d'analyse de l'intention : {e}. Utilisation de valeurs par défaut.")
        return {
            "intent": "definition",
            "general_query": user_query,
            "academic_query": user_query
        }

async def filter_and_rank_sources(user_query: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Étape 3 : Filtre et classe les sources pour ne garder que les plus pertinentes.
    """
    if not sources:
        return []

    source_summaries = "\n".join([f"ID: {i}, Titre: {s['title']}, Snippet: {s.get('snippet', '')}" for i, s in enumerate(sources)])
    prompt = f"""
    Étant donné la question de l'utilisateur et une liste de sources, sélectionne les 3 à 5 sources les plus pertinentes qui répondent le mieux à la question.
    Priorise les sources qui semblent être des définitions directes ou des articles de synthèse de haute qualité.

    Question: "{user_query}"

    Sources:
    {source_summaries}

    Retourne uniquement une liste JSON des ID des sources les plus pertinentes. Exemple : [0, 2, 5]
    """
    try:
        response = await llm_model.generate_content_async(prompt)
        match = re.search(r'\[.*\]', response.text, re.DOTALL)
        if match:
            best_ids = json.loads(match.group(0))
            return [sources[i] for i in best_ids if i < len(sources)]
        return sources[:3] # Fallback
    except Exception as e:
        logger.error(f"Erreur de filtrage des sources : {e}. Utilisation des 3 premières sources.")
        return sources[:3]

async def synthesize_report_tool(user_query: str, research_data: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Étape 5 : Génère la synthèse finale à partir du contexte pertinent."""
    context_str = "\n\n---\n\n".join([f"Source URL: {d['url']}\nContenu:\n{d['content']}" for d in research_data])
    prompt = f"""
    Tu es un assistant de recherche médicale expert. En te basant EXCLUSIVEMENT sur les informations fournies dans le contexte ci-dessous, rédige un rapport clair, structuré et précis en Markdown qui répond à la question de l'utilisateur.

    Si le contexte contient une définition claire, commence par celle-ci. Si le contexte compare des traitements, utilise un tableau.

    Question de l'utilisateur: "{user_query}"

    Contexte fourni:
    {context_str}

    Ton rapport complet en Markdown:
    """
    try:
        stream = await llm_model.generate_content_async(prompt, stream=True)
        async for chunk in stream:
            yield json.dumps({"type": "chunk", "content": chunk.text})
    except Exception as e:
        logger.exception("Erreur lors de la synthèse du rapport.")
        yield json.dumps({"type": "error", "content": f"Erreur lors de la création du rapport : {e}"})

# ==============================================================================
# PARTIE 5: PIPELINE DE L'AGENT ORCHESTRATEUR
# ==============================================================================

async def research_orchestrator_pipeline(query: str, mode: str) -> AsyncGenerator[str, None]:
    """
    La pipeline complète qui orchestre les différentes étapes de la recherche.
    """
    # Étape 1: Analyse de l'intention
    yield json.dumps({"type": "status", "content": "Analyse de votre question..."})
    intent_analysis = await analyze_query_intent(query)
    intent = intent_analysis['intent']
    general_query = intent_analysis['general_query']
    academic_query = intent_analysis['academic_query']
    yield json.dumps({"type": "status", "content": f"Intention détectée : {intent}. Lancement de la recherche..."})

    # Étape 2: Recherche d'informations (dynamique selon l'intention)
    search_tasks = []
    if intent in ['definition', 'treatment', 'comparison']:
        search_tasks.append(web_search_tool(general_query, num_results=5))
    if intent in ['clinical_data', 'treatment', 'comparison'] or mode == 'deep':
        search_tasks.append(academic_search_tool(academic_query, num_results=5))
    
    search_results_lists = await asyncio.gather(*search_tasks)
    all_results = [item for sublist in search_results_lists for item in sublist]

    if not all_results:
        yield json.dumps({"type": "error", "content": "Aucun document trouvé pour cette requête."})
        return

    # Dédoublonnage des résultats
    unique_results = {item['link']: item for item in all_results}.values()
    yield json.dumps({"type": "status", "content": f"{len(unique_results)} sources potentielles trouvées."})

    # Étape 3: Filtrage et classement des sources
    yield json.dumps({"type": "status", "content": "Sélection des sources les plus pertinentes..."})
    best_sources_metadata = await filter_and_rank_sources(query, list(unique_results))
    
    if not best_sources_metadata:
        yield json.dumps({"type": "error", "content": "Impossible de déterminer les sources pertinentes."})
        return
        
    yield json.dumps({"type": "status", "content": f"Extraction du contenu de {len(best_sources_metadata)} sources clés..."})

    # Étape 4: Scraping du contenu des sources pertinentes
    scrape_tasks = [scrape_tool(src['link']) for src in best_sources_metadata]
    scraped_contents = await asyncio.gather(*scrape_tasks)

    research_data = []
    final_sources = []
    for i, content in enumerate(scraped_contents):
        if content and len(content) > 50:
            metadata = best_sources_metadata[i]
            research_data.append({"url": metadata['link'], "title": metadata['title'], "content": content})
            final_sources.append(Source(title=metadata['title'], url=metadata['link'], snippet=metadata.get('snippet')).model_dump())

    if not research_data:
        yield json.dumps({"type": "error", "content": "Le contenu des sources clés n'a pas pu être extrait."})
        return

    # Étape 5: Synthèse finale et streaming
    yield json.dumps({"type": "status", "content": "Génération de la synthèse finale..."})
    async for chunk in synthesize_report_tool(query, research_data):
        yield chunk
        
    yield json.dumps({"type": "sources", "content": final_sources})

# ==============================================================================
# PARTIE 6: ENDPOINTS DE L'API
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Démarrage de l'application Orchestrateur...")
    if not all([settings.GEMINI_API_KEY, settings.SERPER_API_KEY, settings.FIRECRAWL_API_KEY]):
        logger.error("FATAL : Des clés API sont manquantes.")
    else:
        logger.info("Toutes les clés API sont configurées.")

@app.get("/health", status_code=200)
async def health_check():
    return {"status": "ok", "version": app.version}

@app.post("/research-stream")
async def perform_research_stream(request: ResearchRequest):
    logger.info(f"Requête de recherche reçue pour '{request.query}' en mode '{request.mode}'")
    return EventSourceResponse(research_orchestrator_pipeline(request.query, request.mode))

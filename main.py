# ==============================================================================
# FICHIER 1: main.py
# RÔLE: Agent de recherche médicale avancé avec des outils spécialisés
#       et un scraping robuste via Firecrawl.
# LANGUE: Français
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
    """Configure le logging central pour l'application."""
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
    """Charge les configurations et les clés API depuis l'environnement."""
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY")
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY") # Clé pour le scraping robuste

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()
setup_logging()
logger = logging.getLogger(__name__)

# --- Initialisation de l'application FastAPI ---
app = FastAPI(
    title="Agent de Recherche Médicale Avancé",
    description="Un agent IA utilisant des outils spécialisés pour effectuer des recherches médicales et streamer les résultats en temps réel.",
    version="2.1.0" # Version avec scraping amélioré
)

# --- Configuration du Middleware CORS ---
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
    logger.info("Modèle Google Gemini initialisé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors de la configuration de l'API Gemini : {e}")
    llm_model = None

# ==============================================================================
# PARTIE 2: MODÈLES DE DONNÉES (PYDANTIC)
# ==============================================================================

class ResearchRequest(BaseModel):
    """Modèle pour la requête de recherche entrante de l'utilisateur."""
    query: str = Field(..., description="La question médicale de l'utilisateur.")
    mode: Literal["rapid", "deep"] = Field(..., description="Le mode de recherche à utiliser : 'rapide' ou 'approfondi'.")

class Source(BaseModel):
    """Modèle pour une source d'information unique."""
    title: str
    url: str
    snippet: Optional[str] = None

# ==============================================================================
# PARTIE 3: OUTILS DE RECHERCHE SPÉCIALISÉS
# ==============================================================================

async def search_google_scholar(query: str, num_results: int = 3) -> List[Dict[str, Any]]:
    """Effectue une recherche sur Google Scholar via l'API Serper."""
    logger.info(f"Exécution de la recherche Google Scholar pour : '{query}'")
    url = "https://google.serper.dev/scholar"
    payload = json.dumps({"q": query, "num": num_results})
    headers = {'X-API-KEY': settings.SERPER_API_KEY, 'Content-Type': 'application/json'}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json().get('scholar', [])
        except Exception as e:
            logger.error(f"Erreur lors de la recherche Google Scholar : {e}")
            return []

async def search_pubmed(query: str, num_results: int = 3) -> List[Dict[str, Any]]:
    """Simule une recherche sur PubMed via une recherche Google ciblée."""
    logger.info(f"Exécution de la recherche PubMed pour : '{query}'")
    pubmed_query = f"site:pubmed.ncbi.nlm.nih.gov {query}"
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": pubmed_query, "num": num_results})
    headers = {'X-API-KEY': settings.SERPER_API_KEY, 'Content-Type': 'application/json'}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, data=payload, headers=headers, timeout=15)
            response.raise_for_status()
            return response.json().get('organic', [])
        except Exception as e:
            logger.error(f"Erreur lors de la recherche PubMed : {e}")
            return []

async def scrape_with_firecrawl(url: str) -> str:
    """
    Extrait le contenu d'une URL en utilisant l'API Firecrawl pour éviter les blocages.
    Retourne le contenu au format Markdown.
    """
    logger.info(f"Scraping de l'URL avec Firecrawl : {url}")
    api_url = "https://api.firecrawl.dev/v0/scrape"
    headers = {
        "Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {"url": url}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(api_url, json=payload, headers=headers, timeout=45)
            response.raise_for_status()
            scraped_data = response.json()
            if scraped_data.get("success"):
                logger.info(f"Scraping de {url} réussi via Firecrawl.")
                return scraped_data.get("data", {}).get("markdown", "")
            else:
                logger.error(f"Échec du scraping Firecrawl pour {url}: {scraped_data.get('error')}")
                return ""
        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP Firecrawl pour {url}: {e.response.status_code} - {e.response.text}")
            return ""
        except Exception as e:
            logger.exception(f"Erreur inattendue lors du scraping de {url} avec Firecrawl.")
            return ""

# ==============================================================================
# PARTIE 4: LOGIQUE IA ET SYNTHÈSE
# ==============================================================================

async def get_refined_queries(user_query: str) -> Dict[str, str]:
    """Génère des requêtes de recherche optimisées pour différentes bases de données."""
    prompt = f"""
    En te basant sur la question médicale de l'utilisateur, génère des requêtes de recherche optimisées en anglais pour les bases de données suivantes :
    1.  **PubMed**: Une requête technique et précise, ciblant des essais cliniques, des revues ou des méta-analyses. Utilise des termes de type MeSH si possible.
    2.  **Google Scholar**: Une requête un peu plus large, cherchant également des articles, des directives et des citations.

    Question de l'utilisateur : "{user_query}"

    Retourne la réponse uniquement sous forme d'objet JSON au format suivant :
    {{
      "pubmed_query": "...",
      "scholar_query": "..."
    }}
    """
    try:
        response = await llm_model.generate_content_async(prompt)
        json_str = re.search(r'```json\s*([\s\S]*?)\s*```', response.text)
        if json_str:
            return json.loads(json_str.group(1))
        return json.loads(response.text)
    except Exception as e:
        logger.error(f"Erreur lors de l'affinage des requêtes : {e}. Utilisation de la requête originale.")
        return {"pubmed_query": user_query, "scholar_query": user_query}

async def synthesize_final_report(user_query: str, research_data: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
    """Crée un rapport détaillé et structuré à partir des données collectées et le stream."""
    context_str = "\n\n---\n\n".join(
        [f"Source URL: {d['url']}\n\nTitre: {d['title']}\n\nContenu:\n{d['content']}" for d in research_data]
    )
    
    prompt = f"""
    Tu es un expert en recherche médicale et tu rédiges des synthèses détaillées pour des médecins.
    En te basant EXCLUSIVEMENT sur les contenus extraits ci-dessous, crée un rapport complet et bien structuré qui répond à la question suivante.

    **Question de l'utilisateur :**
    {user_query}

    **Structure du rapport :**
    1.  **Résumé (Executive Summary) :** Donne une réponse brève et concise à la question principale.
    2.  **Résultats Détaillés :** Résume les principales conclusions des sources. Si possible, crée un tableau Markdown pour comparer des études, des dosages, des résultats ou d'autres données pertinentes.
    3.  **Discussion et Limites :** Mentionne brièvement s'il y a des données contradictoires ou quelles sont les limites mentionnées dans les sources.
    4.  **Conclusion :** Formule une conclusion finale.

    **IMPORTANT :**
    -   Formate ta réponse entièrement en Markdown.
    -   Sois objectif et tiens-toi strictement aux informations fournies.
    -   N'invente aucune information. Si les données manquent, mentionne-le.

    **Données collectées :**
    {context_str}

    **Ton rapport :**
    """
    try:
        stream = await llm_model.generate_content_async(prompt, stream=True)
        async for chunk in stream:
            yield json.dumps({"type": "chunk", "content": chunk.text})
    except Exception as e:
        logger.exception("Erreur lors de la synthèse finale.")
        yield json.dumps({"type": "error", "content": f"Erreur lors de la création du rapport : {e}"})

# ==============================================================================
# PARTIE 5: PIPELINE DE RECHERCHE (STREAMING)
# ==============================================================================

async def research_pipeline(query: str, mode: str) -> AsyncGenerator[str, None]:
    """La pipeline centrale qui contrôle et streame l'ensemble du processus de recherche."""
    yield json.dumps({"type": "status", "content": "Analyse de la requête..."})
    
    refined_queries = await get_refined_queries(query)
    yield json.dumps({"type": "status", "content": f"Requêtes spécialisées créées (PubMed : '{refined_queries['pubmed_query']}')"})
    
    yield json.dumps({"type": "status", "content": "Lancement de la recherche dans les bases de données médicales..."})
    
    num_results = 5 if mode == 'deep' else 2
    pubmed_task = asyncio.create_task(search_pubmed(refined_queries['pubmed_query'], num_results))
    scholar_task = asyncio.create_task(search_google_scholar(refined_queries['scholar_query'], num_results))
    
    pubmed_results, scholar_results = await asyncio.gather(pubmed_task, scholar_task)
    
    all_results = pubmed_results + scholar_results
    if not all_results:
        yield json.dumps({"type": "error", "content": "Aucun document pertinent trouvé dans les bases de données."})
        return

    unique_sources = {res['link']: Source(title=res.get('title', 'Titre inconnu'), url=res['link'], snippet=res.get('snippet')) for res in all_results if res.get('link')}
    
    sources_list = list(unique_sources.values())
    yield json.dumps({"type": "status", "content": f"{len(sources_list)} sources uniques trouvées."})
    
    yield json.dumps({"type": "status", "content": "Extraction du contenu des sources trouvées..."})
    # **CHANGEMENT** : Utilisation de scrape_with_firecrawl
    scrape_tasks = [asyncio.create_task(scrape_with_firecrawl(src.url)) for src in sources_list]
    scraped_contents = await asyncio.gather(*scrape_tasks)
    
    research_data = []
    valid_sources = []
    for i, content in enumerate(scraped_contents):
        if content and len(content) > 100:
            research_data.append({"url": sources_list[i].url, "title": sources_list[i].title, "content": content})
            valid_sources.append(sources_list[i].model_dump())

    if not research_data:
        yield json.dumps({"type": "error", "content": "Le contenu des sources trouvées n'a pas pu être extrait."})
        return
        
    yield json.dumps({"type": "status", "content": f"Analyse du contenu de {len(research_data)} sources. Lancement de la synthèse..."})

    async for chunk in synthesize_final_report(query, research_data):
        yield chunk
        
    yield json.dumps({"type": "sources", "content": valid_sources})

# ==============================================================================
# PARTIE 6: ENDPOINTS DE L'API
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    """S'exécute au démarrage de l'application."""
    logger.info("Démarrage de l'application...")
    if not all([settings.GEMINI_API_KEY, settings.SERPER_API_KEY, settings.FIRECRAWL_API_KEY]):
        logger.error("FATAL : Des clés API importantes sont manquantes. L'application ne fonctionnera pas correctement.")
    else:
        logger.info("Toutes les clés API requises sont configurées.")

@app.get("/health", status_code=200, tags=["Système"])
async def health_check():
    """Endpoint de santé pour les services de monitoring."""
    return {"status": "ok", "version": app.version}

@app.post("/research-stream", tags=["Recherche"])
async def perform_research_stream(request: ResearchRequest):
    """
    Lance une recherche et streame la progression et les résultats
    via Server-Sent Events (SSE).
    """
    logger.info(f"Requête de recherche en streaming reçue pour '{request.query}' en mode '{request.mode}'")
    
    async def event_generator():
        try:
            async for data_str in research_pipeline(request.query, request.mode):
                yield {"data": data_str}
        except Exception as e:
            logger.exception(f"Une erreur est survenue dans la research_pipeline : {e}")
            error_message = json.dumps({"type": "error", "content": f"Une erreur interne du serveur est survenue : {e}"})
            yield {"data": error_message}

    return EventSourceResponse(event_generator())

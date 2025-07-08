# ==============================================================================
# FICHIER 3: main.py
# RÔLE : Fichier unique contenant TOUTE la logique de l'application.
# Il remplace tous les anciens fichiers du dossier 'app'.
# À PLACER À LA RACINE DE VOTRE PROJET.
# ==============================================================================
import os
import sys
import logging
import asyncio
import json
from typing import List, Optional, Literal

import httpx
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings # CORRECTION : Importation depuis pydantic_settings
from dotenv import load_dotenv

# ==============================================================================
# PARTIE 1: CONFIGURATION ET INITIALISATION
# (Remplace les fichiers dans app/core)
# ==============================================================================

# --- Configuration du Logging ---
def setup_logging():
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(log_formatter)
    root_logger.addHandler(stream_handler)

# --- Gestion des Variables d'Environnement (Secrets) ---
load_dotenv()

class Settings(BaseSettings):
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY")
    SERPER_API_KEY: str = os.getenv("SERPER_API_KEY")
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY")
    
    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

# --- Initialisation de FastAPI et Gemini ---
setup_logging()
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Agent de Recherche Médicale Avancé (Simplifié)",
    description="Un agent IA pour effectuer des recherches médicales rapides ou approfondies.",
    version="1.2.0" # Version mise à jour après correction
)

try:
    genai.configure(api_key=settings.GEMINI_API_KEY)
    llm_model = genai.GenerativeModel('gemini-1.5-flash')
except Exception as e:
    logger.error(f"Erreur lors de la configuration de l'API Gemini : {e}")
    llm_model = None

# ==============================================================================
# PARTIE 2: MODÈLES DE DONNÉES (PYDANTIC)
# (Remplace les fichiers dans app/models)
# ==============================================================================

class ResearchRequest(BaseModel):
    query: str = Field(..., description="La question de l'utilisateur pour la recherche.", min_length=10)
    mode: Literal["rapid", "deep"] = Field(..., description="Le mode de recherche à utiliser.")

class Source(BaseModel):
    title: str
    url: str
    snippet: Optional[str] = None

class ResearchResponse(BaseModel):
    summary: str = Field(..., description="La synthèse générée par l'IA.")
    sources: List[Source] = Field(..., description="La liste des sources utilisées pour la synthèse.")

# ==============================================================================
# PARTIE 3: SERVICES EXTERNES (RECHERCHE, SCRAPING, IA)
# (Remplace les fichiers dans app/services)
# ==============================================================================

# --- Service LLM (Gemini) ---

async def generate_text(prompt: str) -> str:
    if not llm_model:
        raise RuntimeError("Le modèle Gemini n'est pas initialisé.")
    try:
        logger.info("Génération de texte avec Gemini...")
        response = await llm_model.generate_content_async(prompt)
        logger.info("Texte généré avec succès.")
        return response.text
    except Exception as e:
        logger.exception("Erreur lors de l'appel à l'API Gemini.")
        raise

async def generate_json_response(prompt: str, fallback: any):
    response_text = await generate_text(prompt)
    try:
        return json.loads(response_text.strip())
    except json.JSONDecodeError:
        logger.error(f"Échec du décodage JSON de Gemini. Réponse obtenue: {response_text}")
        return fallback

async def generate_search_queries(user_query: str) -> list[str]:
    prompt = f"""
    En te basant sur la question médicale de l'utilisateur, génère 3 requêtes de recherche Google optimisées.
    Cible des sources d'autorité (ex: site:inserm.fr, site:has-sante.fr, site:nejm.org).
    Retourne uniquement une liste JSON de chaînes de caractères.
    Question: "{user_query}"
    Réponse:
    """
    return await generate_json_response(prompt, fallback=[user_query])

async def synthesize_rapid_answer(user_query: str, contexts: list[dict]) -> str:
    context_str = "\n\n---\n\n".join([f"Source URL: {c['url']}\n\nContenu:\n{c['content'][:5000]}" for c in contexts])
    prompt = f"""
    Tu es un assistant de recherche médicale. Basé EXCLUSIVEMENT sur les contextes fournis, rédige une réponse claire et concise à la question.
    Question: "{user_query}"
    Contextes:
    {context_str}
    Réponse synthétique:
    """
    return await generate_text(prompt)

async def decompose_deep_query(user_query: str) -> list[str]:
    prompt = f"""
    Décompose la question de recherche médicale complexe suivante en 3 à 5 sous-questions spécifiques.
    Retourne uniquement une liste JSON de chaînes de caractères.
    Question: "{user_query}"
    Réponse:
    """
    return await generate_json_response(prompt, fallback=[])

async def extract_structured_info(content: str) -> dict:
    prompt = f"""
    Analyse le texte médical et extrais les informations clés en JSON: "study_type", "key_findings" (liste), "main_conclusion".
    Si une info manque, mets la valeur à null.
    Texte:
    ---
    {content[:8000]}
    ---
    JSON structuré:
    """
    return await generate_json_response(prompt, fallback={})

async def synthesize_deep_report(user_query: str, structured_infos: list[dict]) -> str:
    context_str = "\n\n---\n\n".join([json.dumps(info, indent=2, ensure_ascii=False) for info in structured_infos])
    prompt = f"""
    Tu es un expert en synthèses médicales. Basé EXCLUSIVEMENT sur les résumés structurés fournis, rédige un rapport complet répondant à la question.
    Structure ta réponse. Mets en évidence convergences et divergences.
    Question: "{user_query}"
    Résumés:
    {context_str}
    Rapport de recherche approfondi:
    """
    return await generate_text(prompt)

# --- Service de Recherche Web (Serper) ---

async def perform_web_search(query: str, num_results: int = 5) -> list:
    logger.info(f"Exécution de la recherche web pour : '{query}'")
    url = "https://google.serper.dev/search"
    payload = {"q": query, "num": num_results}
    headers = {'X-API-KEY': settings.SERPER_API_KEY, 'Content-Type': 'application/json'}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            search_results = response.json()
            return search_results.get('organic', [])
        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP Serper: {e.response.status_code}")
            return []
        except Exception as e:
            logger.exception("Erreur inattendue dans perform_web_search.")
            return []

# --- Service de Scraping (Firecrawl) ---

async def scrape_url(url: str) -> str:
    logger.info(f"Scraping de l'URL : {url}")
    api_url = "https://api.firecrawl.dev/v0/scrape"
    headers = {"Authorization": f"Bearer {settings.FIRECRAWL_API_KEY}", "Content-Type": "application/json"}
    payload = {"url": url}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(api_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            scraped_data = response.json()
            if scraped_data.get("success"):
                return scraped_data.get("data", {}).get("markdown", "")
            return ""
        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP Firecrawl pour {url}: {e.response.status_code}")
            return ""
        except Exception as e:
            logger.exception(f"Erreur inattendue lors du scraping de {url}.")
            return ""

async def scrape_multiple_urls(urls: list[str]) -> list[dict]:
    tasks = [scrape_url(url) for url in urls]
    contents = await asyncio.gather(*tasks)
    return [{"url": url, "content": content} for url, content in zip(urls, contents) if content]

# ==============================================================================
# PARTIE 4: PIPELINES DE RECHERCHE
# (Remplace app/services/pipeline_service.py)
# ==============================================================================

async def run_rapid_research(query: str) -> ResearchResponse:
    logger.info(f"Démarrage de la recherche RAPIDE pour : '{query}'")
    search_queries = await generate_search_queries(query)
    search_results = await perform_web_search(search_queries[0], num_results=3)
    if not search_results:
        return ResearchResponse(summary="Désolé, je n'ai pas pu effectuer la recherche web.", sources=[])

    urls_to_scrape = [r['link'] for r in search_results]
    scraped_contents = await scrape_multiple_urls(urls_to_scrape)
    if not scraped_contents:
        return ResearchResponse(summary="Désolé, je n'ai pas pu extraire le contenu des pages web.", sources=[])

    summary = await synthesize_rapid_answer(query, scraped_contents)
    sources = [Source(title=r['title'], url=r['link'], snippet=r.get('snippet')) for r in search_results]
    return ResearchResponse(summary=summary, sources=sources)


async def run_deep_research(query: str) -> ResearchResponse:
    logger.info(f"Démarrage de la recherche APPROFONDIE pour : '{query}'")
    sub_questions = await decompose_deep_query(query)
    if not sub_questions:
        sub_questions = [query]

    search_tasks = [perform_web_search(sq, num_results=2) for sq in sub_questions]
    search_results_list = await asyncio.gather(*search_tasks)
    
    all_search_results = []
    all_urls_to_scrape = set()
    for results in search_results_list:
        for r in results:
            if r['link'] not in all_urls_to_scrape:
                all_urls_to_scrape.add(r['link'])
                all_search_results.append(r)
    
    if not all_urls_to_scrape:
        return ResearchResponse(summary="Désolé, la recherche approfondie n'a retourné aucun résultat.", sources=[])

    scraped_contents = await scrape_multiple_urls(list(all_urls_to_scrape))
    extraction_tasks = [extract_structured_info(c['content']) for c in scraped_contents]
    structured_infos = await asyncio.gather(*extraction_tasks)
    
    valid_infos = [info for info in structured_infos if info]
    if not valid_infos:
        return ResearchResponse(summary="Je n'ai pas pu extraire d'informations structurées des documents trouvés.", sources=[])

    summary = await synthesize_deep_report(query, valid_infos)
    sources = [Source(title=r['title'], url=r['link'], snippet=r.get('snippet')) for r in all_search_results]
    return ResearchResponse(summary=summary, sources=sources)

# ==============================================================================
# PARTIE 5: ENDPOINTS DE L'API
# (Contenu original de app/main.py)
# ==============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("Démarrage de l'application...")
    if not all([settings.GEMINI_API_KEY, settings.SERPER_API_KEY, settings.FIRECRAWL_API_KEY]):
        logger.error("FATAL: Clés API manquantes. L'application risque de ne pas fonctionner.")
        # L'application continuera de fonctionner mais les appels API échoueront.
    else:
        logger.info("Toutes les clés API sont configurées.")

@app.get("/health", status_code=200, tags=["Surveillance"])
async def health_check():
    """Endpoint de santé pour la surveillance par Render."""
    return {"status": "ok"}

@app.post("/research", response_model=ResearchResponse, tags=["Recherche"])
async def perform_research(request: ResearchRequest):
    """
    Lance une tâche de recherche en mode 'rapide' ou 'approfondi'.
    """
    logger.info(f"Requête de recherche reçue pour '{request.query}' en mode '{request.mode}'")
    try:
        if request.mode == "rapid":
            result = await run_rapid_research(request.query)
        elif request.mode == "deep":
            result = await run_deep_research(request.query)
        else:
            # Ne devrait jamais arriver grâce à la validation Pydantic
            raise HTTPException(status_code=400, detail="Mode de recherche invalide.")

        logger.info(f"Recherche pour '{request.query}' terminée avec succès")
        return result

    except Exception as e:
        logger.exception(f"Une erreur inattendue est survenue lors de la recherche pour '{request.query}'")
        raise HTTPException(status_code=500, detail=f"Une erreur interne du serveur est survenue: {str(e)}")

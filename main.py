"""wiki-agent : seul point de contact MQTT pour Joshua/Panoramix sur les domaines
"contes" et "news" (et tout autre wiki qui s'enregistrerait de la même façon plus
tard). Contrairement à contes-agent/agent-news auparavant, cet agent ne touche à
AUCUNE base de données interne — il consulte les wikis publiés (contes.caronboulme.fr,
news.caronboulme.fr) via leur API de recherche sémantique déjà exposée
(`/api/wiki/search`), et raisonne en boucle agentique (recherche → éventuelle lecture
de page complète → synthèse) avant de répondre.

contes-agent et agent-news redeviennent des producteurs purs (crawl/consolidation/
génération du wiki) : au démarrage, chacun publie une fois (MQTT, retain=True) une
description de son wiki sur `common/wiki_registry/{agent}` — voir on_registry_update
ci-dessous. Grâce au retain, le wiki-agent reçoit ces descriptions immédiatement à la
connexion, quel que soit l'ordre de démarrage des conteneurs.

Les alias MQTT côté Joshua/Panoramix (`.../contes/request`, `.../news/request`) sont
inchangés — seul le process qui y répond change (voir agent_topics ci-dessous), donc
aucune modification n'est nécessaire côté Joshua/Panoramix.
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from urllib.parse import urlsplit

import httpx
import openai
import uvicorn
from fastapi import FastAPI
from nexus_client import NexusClient

import wiki_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MQTT_HOST            = os.environ["MQTT_HOST"]
MQTT_PORT            = int(os.environ.get("MQTT_PORT", "1883"))
# Authentik OAuth (client credentials) — remplace l'ancienne clé API statique
# MQTT_SERVICE_API_KEY, rejetée par mosquitto-auth-authentik qui exige un vrai JWT
# Authentik (voir _new_nexus_client ci-dessous, même pattern qu'agent-news/main.py).
AUTHENTIK_URL        = os.environ.get("AUTHENTIK_URL", "https://sso.caronboulme.fr")
AUTHENTIK_CLIENT_ID  = os.environ["AUTHENTIK_CLIENT_ID"]
AUTHENTIK_CLIENT_SECRET = os.environ["AUTHENTIK_CLIENT_SECRET"]
LLM_BASE_URL         = os.environ.get("LLM_BASE_URL", "https://thebrain.caronboulme.fr/v1")
LLM_MODEL            = os.environ.get("LLM_MODEL", "qwen3-vl-8b-instruct")
LLAMACPP_API_KEY     = os.environ["LLAMACPP_API_KEY"]

# Budget fixe d'étapes d'exploration (structure + lecture(s)). Plus d'extension selon
# la longueur des pages : constaté le 2026-10-18, une catégorie de 355 000 caractères
# faisait passer le budget de 11 à 41 étapes et monopolisait le GPU partagé.
MAX_TOOL_ITERATIONS = 10
# Panoramix abandonne l'attente à 60 s (mqtt_write.DEFAULT_RESPONSE_TIMEOUT) :
# au-delà, continuer ne sert à personne. On rend ce qui a été lu à temps.
EXPLORATION_DEADLINE_SECONDS = 50

# wiki enregistrés par les agents producteurs (contes-agent, agent-news...), tenu à
# jour par on_registry_update — clé = "agent" du message publié sur common/wiki_registry/*
# (ex: "contes", "news"), valeur = {"agent", "base_url", "search_path", "description"}.
_registry: dict[str, dict] = {}
_subscribed_sessions: set[str] = set()

app = FastAPI(title="wiki-agent")


@app.get("/health")
async def health():
    return {"status": "ok", "wikis": list(_registry.keys())}


def _get_llm_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=LLAMACPP_API_KEY, base_url=LLM_BASE_URL)


async def _new_nexus_client() -> NexusClient:
    """Obtient un token OAuth Client Credentials via Authentik et construit un
    NexusClient authentifié — mosquitto-auth-authentik valide le mot de passe MQTT
    comme un JWT Authentik, un token frais est donc nécessaire à chaque connexion."""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{AUTHENTIK_URL}/application/o/token/",
            data={
                "grant_type": "client_credentials",
                "client_id": AUTHENTIK_CLIENT_ID,
                "client_secret": AUTHENTIK_CLIENT_SECRET,
            },
        )
        if resp.status_code != 200:
            logger.error(f"Échec obtention token OAuth: {resp.status_code} {resp.text}")
            raise RuntimeError("Cannot get OAuth token")
        access_token = resp.json()["access_token"]

    return await NexusClient.from_authentik_token(
        AUTHENTIK_URL, MQTT_HOST, access_token,
        AUTHENTIK_CLIENT_ID, AUTHENTIK_CLIENT_SECRET, MQTT_PORT,
    )


# ---------------------------------------------------------------------------
# Boucle agentique
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_wiki",
            "description": (
                "Recherche sémantique dans le wiki — renvoie les pages les plus "
                "pertinentes pour un sujet libre (titre, résumé, score, url). ÉTAPE "
                "DE DÉCOUVERTE UNIQUEMENT : le résumé renvoyé est un extrait tronqué "
                "d'index, jamais le contenu complet — n'utilise JAMAIS ce résumé "
                "comme matière finale, il sert seulement à repérer quelles URLs lire "
                "ensuite avec fetch_wiki_page (obligatoire pour toute page jugée "
                "pertinente)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Termes de recherche (peut différer de la question originale pour affiner)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_wiki_page",
            "description": (
                "Récupère le contenu d'une URL du wiki — un index de structure JSON "
                "(voir la structure du wiki dans le message système : point de départ "
                "'index', champs 'index'/'page' à l'intérieur pour continuer "
                "l'exploration), une page éditoriale trouvée comme champ 'page', ou "
                "une URL renvoyée par search_wiki. OBLIGATOIRE avant de considérer un "
                "résultat de search_wiki comme utilisable. Une page longue (ex: la "
                "transcription complète d'une histoire) peut dépasser la taille d'un "
                "seul appel : la réponse l'indique alors avec un offset à réutiliser "
                "pour continuer la lecture — enchaîne ces appels si la question porte "
                "sur un passage précis qui n'apparaît pas encore."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL absolue à récupérer."},
                    "offset": {
                        "type": "integer",
                        "description": (
                            "Position (en caractères) à partir de laquelle continuer la "
                            "lecture d'une page déjà partiellement lue — voir la note de "
                            "continuation dans une réponse précédente. Omettre pour "
                            "commencer au début."
                        ),
                    },
                },
                "required": ["url"],
            },
        },
    },
]


def _next_exploration_step(llm_client: openai.OpenAI, model: str, messages: list[dict]) -> list[dict]:
    """Une étape d'exploration : renvoie les appels d'outils demandés par le LLM, ou []
    s'il considère avoir assez de matière.

    Réponse lue en streaming : dès que le modèle produit du texte au lieu d'un appel
    d'outil, la décision d'arrêt est prise et on coupe le flux (vLLM annule alors la
    génération). Sans ça, le modèle rédigeait souvent une réponse complète (observé :
    bulletin de 659 tokens, 22 s) que _gather_from_wiki jette de toute façon — seul
    compte le fait qu'il n'appelle plus d'outil. Sur 261 réponses avec appel d'outil
    observées, aucune ne commençait par du texte : couper au premier texte est sûr."""
    stream = llm_client.chat.completions.create(
        model=model, messages=messages, tools=_TOOLS, tool_choice="auto", timeout=60, stream=True,
    )
    calls: dict[int, dict] = {}
    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = calls.setdefault(tc.index, {"id": None, "name": "", "arguments": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
            elif delta.content and delta.content.strip() and not calls:
                logger.info("  fin d'exploration (le modèle rédige au lieu d'appeler un outil) — génération coupée")
                return []
    finally:
        stream.close()
    result = []
    for index, c in sorted(calls.items()):
        if c["name"]:
            c["id"] = c["id"] or f"call_{index}"
            result.append(c)
    return result


_TZ = ZoneInfo(os.environ.get("TZ_USER", "Europe/Paris"))
_DAYS_FR = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]
_MONTHS_FR = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
              "août", "septembre", "octobre", "novembre", "décembre"]


def _today_fr() -> str:
    """Date du jour en français, à l'heure de Paris (été/hiver compris) : le
    conteneur est en UTC, et strftime y donne les noms anglais."""
    now = datetime.now(_TZ)
    day = "1er" if now.day == 1 else str(now.day)
    return f"{_DAYS_FR[now.weekday()]} {day} {_MONTHS_FR[now.month - 1]} {now.year}"


async def _gather_from_wiki(domain: str, query: str, llm_client: openai.OpenAI, model: str) -> str:
    """Explore le wiki (search_wiki + fetch_wiki_page) pour rassembler la matière brute
    pertinente à la question — ne rédige PAS de réponse : c'est le rôle de l'assistant
    appelant (Joshua/Panoramix), qui a le contexte conversationnel complet (ton, historique,
    profil utilisateur). Faire rédiger une réponse ici doublerait inutilement l'aller-retour
    LLM (latence) et remplacerait la matière brute par un résumé, perdant le détail que
    l'assistant appelant pourrait juger pertinent (ex: une URL source précise demandée
    ensuite).

    search_wiki n'est qu'une étape de DÉCOUVERTE de pages pertinentes, jamais une source
    de contenu : ses résultats ne sont JAMAIS ajoutés à `collected` (voir la boucle
    ci-dessous) — seul le contenu réellement lu via fetch_wiki_page l'est. Contrainte
    imposée au niveau du code plutôt que seulement au LLM par prompt, pour qu'elle tienne
    même si le modèle ne la respecte pas spontanément."""
    entry = _registry.get(domain)
    if not entry:
        return f"Le wiki '{domain}' n'est pas encore disponible."

    today = _today_fr()
    structure = entry.get("structure")
    structure_hint = f"\n\nStructure du wiki : {structure}" if structure else ""
    system = (
        f"Tu rassembles la matière brute pertinente à une question sur "
        f"{entry.get('description', domain)}, en explorant le wiki publié sur "
        f"{entry['base_url']}.\n\n"
        "Deux façons de trouver des pages pertinentes, à combiner selon la question :\n"
        "1. Pars de l'index de structure (voir ci-dessous), et suis les champs 'index' "
        "(sous-sections à explorer) et 'page' (contenu à lire) — la meilleure approche "
        "pour une question générale ou bornée dans le temps (\"aujourd'hui\", \"la "
        "semaine dernière\"), qui donne accès au contenu déjà organisé/rédigé (ex: le "
        "bulletin du jour).\n"
        "2. Pour un sujet précis nommé, search_wiki peut aider à repérer des pages — "
        "mais son résumé n'est qu'un extrait tronqué de l'index de recherche, JAMAIS "
        "une source de contenu utilisable telle quelle : toute page qui ressort d'une "
        "recherche et semble pertinente doit ENSUITE être lue avec fetch_wiki_page "
        "avant d'être considérée comme couvrant la question.\n\n"
        "search_wiki peut renvoyer, pour une même page longue (ex: un conte), à la fois "
        "son URL générale et des URLs avec une ancre (#periode-N, #section-N...) "
        "pointant chacune un passage précis. Si la question porte sur un événement, un "
        "moment ou un détail précis (pas le résumé général de la page), privilégie "
        "TOUJOURS l'URL la mieux notée (score le plus élevé) parmi celles avec ancre "
        "plutôt que l'URL générale sans ancre — l'ancre place directement "
        "fetch_wiki_page sur le passage pertinent au lieu de démarrer au début d'une "
        "page qui peut faire plusieurs centaines de milliers de caractères. Si la "
        "première page lue ne répond pas à la question, n'abandonne pas : essaie une "
        "autre ancre pertinente parmi les résultats de recherche, ou pagine plus loin "
        "dans la même page avec l'offset indiqué (\"restants\").\n\n"
        "Une page lue via fetch_wiki_page inclut les liens qu'elle contient sous la "
        "forme 'texte (url)'. Si la question porte sur les sources/URLs d'un article "
        "(pas juste son contenu), va jusqu'à la fiche sujet correspondante et prends "
        "l'URL qui suit 'Sources :' dans son contenu, c'est la seule vraie URL "
        f"d'article — jamais une URL de ce wiki lui-même ({entry['base_url']}). "
        "N'appelle plus aucun outil dès que tu as rassemblé de quoi couvrir la "
        "question : tu ne rédiges PAS de réponse toi-même, quelqu'un d'autre s'en "
        f"charge à partir de ce que tu auras trouvé.{structure_hint}"
        # En dernier : seule partie qui change (chaque jour), pour le cache de préfixe
        f"\n\nNous sommes le {today}."
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": query},
    ]

    collected: list[str] = []
    # Index JSON de structure (…/index.json) : utiles au LLM pour naviguer, mais pas
    # renvoyés à l'appelant tant qu'une vraie page de contenu a été lue — ex: la liste
    # des dates (12 000 caractères) doublait la taille du résultat d'une question
    # d'actualité, que l'assistant appelant devait relire (~4 s) avant de répondre.
    collected_navigation: list[str] = []
    loop = asyncio.get_event_loop()
    deadline = time.monotonic() + EXPLORATION_DEADLINE_SECONDS
    iteration = 0
    while iteration < MAX_TOOL_ITERATIONS:
        if time.monotonic() > deadline:
            logger.warning(f"_gather_from_wiki({domain}, {query!r}): délai de {EXPLORATION_DEADLINE_SECONDS}s atteint après {iteration} étape(s), on rend ce qui a été lu")
            break
        iteration += 1
        tool_calls = await loop.run_in_executor(
            None, lambda: _next_exploration_step(llm_client, model, messages)
        )
        if not tool_calls:
            break

        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": tc["id"], "type": "function",
                 "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                for tc in tool_calls
            ],
        })
        for tc in tool_calls:
            try:
                args = json.loads(tc["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            if tc["name"] == "search_wiki":
                q = args.get("query", query)
                results = await wiki_client.search_wiki(entry["base_url"], entry["search_path"], q)
                result_text = json.dumps(results, ensure_ascii=False)
                logger.info(f"  search_wiki({q!r}) -> {len(results)} résultat(s)")
                # Volontairement PAS ajouté à `collected` : simple étape de découverte,
                # jamais restitué tel quel (voir docstring de _gather_from_wiki) — le
                # LLM voit le résultat pour décider quoi fetch_wiki_page ensuite, mais
                # il ne peut jamais devenir la réponse finale par ce chemin.
            elif tc["name"] == "fetch_wiki_page":
                url = args.get("url", "")
                offset = int(args.get("offset") or 0)
                result_text, remaining_chars = await wiki_client.fetch_wiki_page(url, offset)
                logger.info(f"  fetch_wiki_page({url!r}, offset={offset}) -> {len(result_text)} chars, {remaining_chars} restants")
                page = f"[Contenu récupéré du wiki — pas une source d'article]\n{result_text}"
                if urlsplit(url).path.endswith(".json"):
                    collected_navigation.append(page)
                else:
                    collected.append(page)
            else:
                result_text = f"Outil inconnu: {tc['name']}"
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result_text})
    else:
        logger.warning(f"_gather_from_wiki({domain}, {query!r}): boucle interrompue après {MAX_TOOL_ITERATIONS} étapes")

    if not collected:
        # Aucune page de contenu lue : l'index peut être la réponse elle-même
        # (ex: "quels contes as-tu ?" → liste des items d'un index)
        collected = collected_navigation
    if not collected:
        return "Rien de pertinent trouvé sur le wiki pour cette question."
    return "\n\n---\n\n".join(collected)


# ---------------------------------------------------------------------------
# MQTT
# ---------------------------------------------------------------------------

async def on_registry_update(topic: str, payload) -> None:
    if not isinstance(payload, dict):
        return
    agent = payload.get("agent")
    base_url = payload.get("base_url")
    search_path = payload.get("search_path")
    if not agent or not base_url or not search_path:
        logger.warning(f"Message wiki/registry incomplet, ignoré: {payload!r}")
        return
    _registry[agent] = payload
    logger.info(f"Wiki enregistré: {agent!r} -> {base_url}{search_path}")


async def on_user_connected(topic: str, payload) -> None:
    if not isinstance(payload, dict):
        return

    username = payload.get("username")
    authenticated = payload.get("authenticated")
    session_id = payload.get("session_id")
    private_topics = payload.get("private_topics", [])

    if not username or not authenticated or not session_id:
        return

    agent_topics_topic = None
    for entry in private_topics:
        for t in entry.get("topics", []):
            if t["topic"].endswith("/agent_topics"):
                agent_topics_topic = t["topic"]
                break

    if not agent_topics_topic:
        logger.warning(f"[{username}] agent_topics introuvable, skip")
        return

    if not _registry:
        logger.warning(f"[{username}] Aucun wiki enregistré pour l'instant, rien à annoncer")
        return

    nexus = await _new_nexus_client()

    manifest = []
    domain_topics: dict[str, tuple[str, str]] = {}
    for domain, entry in _registry.items():
        request_topic = f"users/{username}/{session_id}/{domain}/request"
        result_topic = f"users/{username}/{session_id}/{domain}/result"
        domain_topics[domain] = (request_topic, result_topic)
        manifest.append({
            "agent": domain,
            "topics": [
                {
                    "topic": request_topic,
                    "description": (
                        f"Consultation du wiki {domain} : {entry.get('description', '')} "
                        "Champ 'query' obligatoire (texte libre : nom, thème, question précise)."
                    ),
                    "access": "write",
                    "response_topic": result_topic,
                    "slow": True,  # réponse en plusieurs secondes : phrase d'attente
                    "format": {"query": "texte libre de la question"},
                },
                {
                    "topic": result_topic,
                    "description": "Réponse. Champ 'content' contient le texte.",
                    "access": "read",
                    "format": {"type": "string", "content": "string"},
                },
            ],
        })

    await nexus.publish(agent_topics_topic, manifest)
    logger.info(f"[{username}/{session_id}] Topics {list(_registry.keys())} déclarés")

    session_key = (username, session_id)
    if session_key in _subscribed_sessions:
        return
    _subscribed_sessions.add(session_key)

    def _make_handler(domain: str, result_topic: str):
        async def on_request(t: str, p) -> None:
            if not isinstance(p, dict):
                logger.warning(f"[{username}] Requête {domain} ignorée: payload non-dict: {p!r}")
                await nexus.publish(result_topic, {"error": "invalid request payload: expected a JSON object"})
                return
            reply_to = p.get("reply_to", result_topic)
            query = (p.get("query") or p.get("topic") or "").strip()
            if not query:
                await nexus.publish(reply_to, {"type": domain, "content": "Précise ta question (champ 'query')."})
                return
            logger.info(f"[{username}] Requête {domain}: {query!r}")
            try:
                llm = _get_llm_client()
                content = await _gather_from_wiki(domain, query, llm, LLM_MODEL)
            except Exception as e:
                # nexus_client._on_message dispatche les callbacks coroutine via
                # asyncio.run_coroutine_threadsafe sans jamais attendre le résultat :
                # toute exception non catchée ici laisserait l'appelant bloqué
                # indéfiniment sur reply_to (même précaution que contes-agent/agent-news).
                logger.exception(f"[{username}] Erreur traitement requête {domain}: {e}")
                content = f"Erreur interne : {e}"
                await nexus.publish(reply_to, {"type": domain, "content": content})
                return
            logger.info(f"[{username}] Réponse {domain} ({len(content)} chars)")
            await nexus.publish(reply_to, {"type": domain, "content": content})
        return on_request

    for domain, (request_topic, result_topic) in domain_topics.items():
        nexus.subscribe(request_topic, _make_handler(domain, result_topic))
    nexus.start_listening()
    logger.info(f"[{username}/{session_id}] Abonné à {list(domain_topics.keys())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    nexus = await _new_nexus_client()
    nexus.subscribe("common/wiki_registry/#", on_registry_update)
    nexus.subscribe("common/user_connected", on_user_connected)
    nexus.start_listening()
    logger.info(f"wiki-agent démarré — LLM: {LLM_MODEL}")

    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    asyncio.run(main())

"""Client HTTP vers les wikis publiés (contes.caronboulme.fr, news.caronboulme.fr) —
le wiki-agent ne touche à aucune base de données interne des agents producteurs
(contes-agent, agent-news), seulement à leur contenu publié, via les mêmes endpoints
que les navigateurs qui consultent le wiki (voir wiki_api.py de chaque repo).

search_wiki (recherche sémantique) n'est qu'une étape de DÉCOUVERTE de pages
pertinentes — jamais une source de contenu en soi. Voir main.py::_gather_from_wiki :
ses résultats ne sont jamais ajoutés à la matière renvoyée à l'appelant, seul le
contenu réellement lu via fetch_wiki_page l'est.

Les deux wikis exposent le même chemin (`/api/wiki/search`) mais des formes de réponse
différentes (contes-agent : stories + moments avec des noms de champs distincts pour
le résumé ; agent-news : subjects) — normalisées ici en une seule forme commune pour
que la boucle agentique (main.py) n'ait pas à connaître ces différences.
"""

import logging

import aiohttp

logger = logging.getLogger(__name__)

# 60s plutôt que quelques secondes : constaté sur données réelles, la toute première
# recherche après un redémarrage d'agent-news/contes-agent déclenche le chargement à
# la demande de leur modèle d'embedding (bge-m3), qui inclut plusieurs vérifications
# HTTP vers HuggingFace Hub avant de répondre — largement > 20s la première fois,
# quasi instantané ensuite (modèle chargé en mémoire).

HTTP_TIMEOUT = aiohttp.ClientTimeout(total=60)

# Taille max du texte extrait d'une page (fetch_wiki_page) — une fiche sujet/histoire
# complète avec historique d'éditions peut être longue ; on garde large mais borné pour
# ne pas exploser le contexte du LLM interne du wiki-agent (voir aussi la priorité
# contenu principal / blocs repliables dans _extract_main_text, qui rend cette
# troncature bien moins dommageable qu'une simple coupe en aveugle).
MAX_PAGE_CHARS = 12000


async def search_wiki(base_url: str, search_path: str, query: str, limit: int = 8) -> list[dict]:
    """Recherche sémantique sur un wiki, résultat normalisé :
    [{"title", "summary", "score", "url"}, ...] — url déjà absolue. Sert UNIQUEMENT à
    identifier quelles pages lire ensuite via fetch_wiki_page ; le "summary" retourné
    ici est un extrait tronqué de l'index de recherche, pas le contenu complet de la
    page, et ne doit jamais être restitué tel quel comme réponse."""
    url = f"{base_url}{search_path}"
    try:
        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as session:
            async with session.get(url, params={"q": query, "limit": limit}) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except Exception as e:
        logger.error(f"search_wiki({base_url}, {query!r}) échoué: {e}")
        return []

    results: list[dict] = []
    for story in data.get("stories", []):
        results.append(_normalize(story, base_url, summary_key="summary"))
    for moment in data.get("moments", []):
        results.append(_normalize(moment, base_url, summary_key="excerpt"))
    for subject in data.get("subjects", []):
        results.append(_normalize(subject, base_url, summary_key="summary"))
    return results


def _normalize(item: dict, base_url: str, summary_key: str) -> dict:
    url = item.get("url") or ""
    if url.startswith("/"):
        url = base_url.rstrip("/") + url
    return {
        "title": item.get("title", ""),
        "summary": item.get(summary_key, ""),
        "score": item.get("score", 0.0),
        "url": url,
    }


async def fetch_wiki_page(url: str, offset: int = 0) -> tuple[str, int]:
    """Récupère une page du wiki (URL absolue) — page éditoriale (HTML, trouvée comme
    champ 'page' dans un index JSON) ou index de structure (JSON, voir 'structure' du
    wiki_registry publié par l'agent producteur). Un seul outil pour les deux : le
    contenu-type de la réponse tranche entre extraction de texte HTML et JSON tel quel.

    Renvoie (texte, caractères_restants) — `offset` pagine une page HTML trop longue
    pour tenir en un seul appel (ex: la transcription complète d'un conte, jusqu'à
    240 000+ caractères constatés). Le nombre de caractères restants (2e élément)
    permet à l'appelant (voir main.py::_gather_from_wiki) d'ajuster son budget
    d'itérations à la taille réelle du contenu plutôt que d'utiliser un plafond fixe
    qui ne scale pas avec la taille de la page.

    Si `url` contient une ancre (#periode-N, ex: renvoyée par search_wiki pour un
    'moment' précis d'un conte) et qu'aucun `offset` explicite n'est fourni, la
    pagination démarre directement à cette ancre (avec un peu de contexte avant) plutôt
    que du tout début de la page — sinon l'ancre, ignorée par un simple fetch HTML (les
    fragments ne sont jamais envoyés au serveur), ne servait à rien : le passage précis
    identifié par la recherche sémantique était perdu, noyé dans le début de la page."""
    from urllib.parse import urlsplit

    fragment = urlsplit(url).fragment
    try:
        async with aiohttp.ClientSession(timeout=HTTP_TIMEOUT) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                content_type = resp.content_type
                body = await resp.text()
    except Exception as e:
        logger.error(f"fetch_wiki_page({url}) échoué: {e}")
        return f"(page inaccessible: {e})", 0

    if content_type == "application/json" or url.endswith(".json"):
        full_text, anchor_offset = body, None
    else:
        full_text, anchor_offset = _extract_main_text(body, base_url=url, fragment=fragment)

    if offset == 0 and anchor_offset is not None:
        offset = max(0, anchor_offset - 300)  # un peu de contexte avant l'ancre, pas juste à partir d'elle
    return _paginate(full_text, offset)


def _paginate(text: str, offset: int) -> tuple[str, int]:
    chunk = text[offset:offset + MAX_PAGE_CHARS]
    end = offset + len(chunk)
    remaining = max(0, len(text) - end)
    if remaining:
        chunk += (
            f"\n\n[... {remaining} caractère(s) restants — rappelle "
            f"fetch_wiki_page avec offset={end} pour continuer la lecture si "
            "cette page ne répond pas encore à la question ...]"
        )
    return chunk, remaining


_ANCHOR_MARKER = "\x00WIKI_ANCHOR\x00"


def _extract_main_text(html: str, base_url: str, fragment: str = "") -> tuple[str, int | None]:
    """Texte principal complet (non tronqué ici, voir _paginate) + liens préservés —
    en donnant TOUJOURS priorité au contenu principal sur les blocs repliables
    (admonitions "??? note", ex: texte intégral d'un conte, analyse détaillée d'un
    sujet d'actualité) : ceux-ci peuvent être énormes et écrasaient jusqu'ici la
    quasi-totalité du contenu utile (résumé, infos, chapitres...) sous une simple
    troncature aveugle — moins de 5% du texte réel conservé dans le pire cas observé
    avant ce fix. Le contenu principal est donc toujours placé avant les blocs
    repliables dans le texte renvoyé.

    Renvoie (texte, position_ancre) — position_ancre est la position dans le texte
    final de l'élément #{fragment} s'il existe sur la page, sinon None. Repéré en
    insérant un marqueur juste avant l'élément ciblé, avant l'aplatissement en texte
    (seul moyen fiable de faire correspondre une position HTML à une position dans le
    texte final, après suppression des balises/liens/blocs repliables)."""
    from urllib.parse import urljoin

    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    main = soup.select_one("article") or soup.select_one(".md-content") or soup.body or soup
    for tag in main.select("nav, header, footer, script, style, .md-sidebar"):
        tag.decompose()

    marked = False
    if fragment:
        target = main.select_one(f"#{fragment}")
        if target is not None:
            target.insert_before(_ANCHOR_MARKER)
            marked = True

    # get_text() jette les href — sans ça, un agent qui lit une fiche sujet voit les
    # noms des sources citées mais jamais leurs URLs, ni les liens vers d'autres pages
    # du wiki (sujet suivant, catégorie...), impossible ensuite d'aller "jusqu'à
    # l'article d'origine". On remplace chaque <a href> par 'texte (url)' avant
    # extraction (URL toujours absolue, résolue depuis la page courante — les liens
    # internes du wiki sont relatifs), seule façon de faire survivre le lien à get_text() ;
    # fait sur tout `main` AVANT de séparer les blocs repliables ci-dessous, pour que les
    # liens survivent dans les deux parts.
    for a in main.select("a[href]"):
        href = a["href"]
        if href.startswith("#"):
            continue
        a.replace_with(f"{a.get_text(strip=True)} ({urljoin(base_url, href)})")

    details = main.select("details")
    details_text = "\n\n".join(d.get_text(separator="\n", strip=True) for d in details)
    for d in details:
        d.decompose()

    primary = main.get_text(separator="\n", strip=True)
    full = f"{primary}\n\n{details_text}" if details_text else primary

    anchor_offset = None
    if marked and _ANCHOR_MARKER in full:
        anchor_offset = full.index(_ANCHOR_MARKER)
        full = full.replace(_ANCHOR_MARKER, "", 1)

    return full, anchor_offset

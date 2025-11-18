from bs4 import BeautifulSoup, Tag
from typing import Tuple, List, Optional

BAD_CONTAINER_KEYWORDS = [
    "nav", "menu", "footer", "header", "sidebar",
    "aside", "ads", "advert", "promo", "related",
    "card", "cards", "teaser", "grid", "carousel"
]


def _score_node(node: Tag) -> Tuple[int, float]:
    """
    Simple heuristic:
    - score by text length
    - penalize nodes that are mostly links / buttons
    """
    text = node.get_text(" ", strip=True)
    text_len = len(text)

    if text_len < 200:  # too small => not main content
        return 0, 0.0

    links = node.find_all("a")
    buttons = node.find_all("button")
    link_like = len(links) + len(buttons)

    # ratio of text to clickable elements
    density = text_len / (link_like + 1)

    return text_len, density


def _looks_like_noise(node: Tag) -> bool:
    """
    Filter out obvious non-content containers by id/class keywords.
    """
    parts: List[str] = []
    if node.get("id"):
        parts.append(str(node.get("id")))
    if node.get("class"):
        parts.extend([str(c) for c in node.get("class")])

    id_class = " ".join(parts).lower()
    return any(bad in id_class for bad in BAD_CONTAINER_KEYWORDS)


def _find_main_content_node(soup: BeautifulSoup) -> Optional[Tag]:
    """
    Try to find the main article-like node:
    - Prefer <article> and <main>
    - Fallback to big <section>/<div>
    """
    candidates: List[Tag] = []

    # Prefer semantic containers first
    for selector in ["article", "main", "section", "div"]:
        for node in soup.select(selector):
            if _looks_like_noise(node):
                continue
            text_len, density = _score_node(node)
            if text_len == 0:
                continue
            # basic threshold to avoid grids of small cards
            if density < 50:  # too many links vs text
                continue
            node._score_len = text_len  # type: ignore[attr-defined]
            node._score_density = density  # type: ignore[attr-defined]
            candidates.append(node)

        if candidates:
            break  # if we found good article/main/section, no need to go down to generic divs

    if not candidates:
        return soup.body or soup

    # pick the node with largest text_len, tie-breaker by density
    candidates.sort(
        key=lambda n: (getattr(n, "_score_len"), getattr(n, "_score_density")),
        reverse=True,
    )
    return candidates[0]


def _extract_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # Remove obvious non-content tags globally
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside", "form"]):
        tag.decompose()

    main_node = _find_main_content_node(soup)

    text = main_node.get_text(separator="\n") if main_node else soup.get_text(separator="\n")

    # Clean empty lines / noise
    lines = (line.strip() for line in text.splitlines())
    chunks = [line for line in lines if line]

    return "\n".join(chunks)
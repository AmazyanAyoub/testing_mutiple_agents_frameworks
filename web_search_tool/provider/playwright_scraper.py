import asyncio
from typing import Any, Dict, List, Optional

from playwright.async_api import async_playwright


async def _fetch_single(
    browser,
    url: str,
    *,
    timeout_ms: int,
    wait_until: str,
    block_heavy: bool,
) -> Dict[str, Any]:
    if not url or not isinstance(url, str) or not url.strip():
        return {"url": url, "status": None, "error": "url must not be empty", "html": ""}

    page = await browser.new_page()

    if block_heavy:
        async def _block(route, request):
            if request.resource_type in {"image", "media", "font", "stylesheet"}:
                await route.abort()
            else:
                await route.continue_()
        await page.route("**/*", _block)

    try:
        response = await page.goto(url, wait_until=wait_until, timeout=timeout_ms)
        html = await page.content()
        status = response.status if response is not None else None
        return {
            "url": url,
            "status": status,
            "error": None if status and 200 <= status < 400 else f"HTTP {status}",
            "html": html or "",
        }
    except Exception as exc:
        return {"url": url, "status": None, "error": str(exc), "html": ""}
    finally:
        await page.close()


async def fetch_pages_html(
    urls: List[str],
    *,
    timeout_ms: int = 30000,
    wait_until: str = "networkidle",
    block_heavy: bool = True,
    max_concurrency: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch multiple pages in parallel using a shared browser instance.
    Returns a list of dicts with url, status, error, html.
    """
    if not urls:
        return []

    semaphore = asyncio.Semaphore(max_concurrency)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            async def _task(u: str) -> Dict[str, Any]:
                async with semaphore:
                    return await _fetch_single(
                        browser,
                        u,
                        timeout_ms=timeout_ms,
                        wait_until=wait_until,
                        block_heavy=block_heavy,
                    )

            return await asyncio.gather(*[_task(u) for u in urls])
        finally:
            await browser.close()


# Backward-compatible single-page fetch, if you still need it
async def fetch_page_html(
    url: str,
    *,
    timeout_ms: int = 30000,
    wait_until: str = "networkidle",
    block_heavy: bool = True,
) -> Dict[str, Any]:
    results = await fetch_pages_html(
        [url],
        timeout_ms=timeout_ms,
        wait_until=wait_until,
        block_heavy=block_heavy,
        max_concurrency=20,
    )
    return results[0] if results else {"url": url, "status": None, "error": "no result", "html": ""}

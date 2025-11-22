import asyncio
from typing import Any, Dict, Optional

from playwright.async_api import async_playwright

async def fetch_page_html(
    url: str,
    *,
    timeout_ms: int = 30000,
    wait_until: str = "networkidle",
    block_heavy: bool = True,
) -> Dict[str, Any]:
    """
    Fetch the raw HTML of a page using Playwright (Chromium).

    Returns a dict with url, status (int|None), error (str|None), html (str).
    No HTML cleaning is done here.
    """
    if not url or not isinstance(url, str) or not url.strip():
        return {"url": url, "status": None, "error": "url must not be empty", "html": ""}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        try:
            page = await browser.new_page()

            if block_heavy:
                async def _block(route, request):
                    if request.resource_type in {"image", "media", "font", "stylesheet"}:
                        await route.abort()
                    else:
                        await route.continue_()
                await page.route("**/*", _block)

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
            await browser.close()

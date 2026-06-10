from __future__ import annotations

import socket
from pathlib import Path
from typing import Any
from urllib import robotparser
from urllib.parse import urljoin, urlparse

from .models import ChunkShotCapture, PageData, QuoteMatch, ViewportChunkData
from .utils import (
    clean_page_text,
    clean_text,
    dom_node_text,
    evidence_quote_candidates,
    filename_for_url,
    find_quote_match,
    is_public_http_url,
    safe_error,
    split_node_indexes_by_text_length,
)


BLOCK_STATUS_TEXT = {
    401: "login_required",
    402: "paywall",
    403: "blocked",
    407: "blocked",
    408: "timeout",
    429: "rate_limited",
}
BLOCK_TEXT_PATTERNS = [
    ("captcha", ["captcha", "recaptcha", "人机验证", "验证码", "安全验证"]),
    ("blocked", ["access denied", "forbidden", "请求被拒绝", "访问被拒绝", "cloudflare"]),
    ("login_required", ["sign in", "log in", "登录", "请先登录", "账号登录"]),
    ("paywall", ["subscribe", "subscription", "付费阅读", "会员专享", "购买会员"]),
]
HARD_BLOCK_TEXT_PATTERNS = [
    ("captcha", ["captcha", "recaptcha", "human verification", "security verification", "人机验证", "验证码", "安全验证"]),
    ("blocked", ["access denied", "forbidden", "request blocked", "cloudflare", "访问被拒绝", "请求被拒绝"]),
]
SOFT_BLOCK_TEXT_PATTERNS = [
    ("login_required", ["sign in", "log in", "login", "登录", "请先登录", "账号登录", "注册/登录"]),
    ("paywall", ["subscribe", "subscription", "paywall", "付费阅读", "会员专享", "购买会员", "订阅后继续阅读"]),
]
BLOCK_TEXT_PATTERNS = HARD_BLOCK_TEXT_PATTERNS + SOFT_BLOCK_TEXT_PATTERNS
LOGIN_URL_HINTS = {"/login", "/signin", "/sign-in", "/account", "/passport", "/user/login"}
PAYWALL_TITLE_HINTS = {"subscribe", "subscription", "paywall", "会员", "订阅", "付费阅读"}


class Screenshotter:
    def __init__(
        self,
        run_dir: Path,
        full_page: bool,
        *,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        disable_proxy: bool = True,
    ):
        self.run_dir = run_dir
        self.screenshot_dir = run_dir / "screenshots"
        self.content_dir = run_dir / "contents"
        self.full_page = full_page
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.disable_proxy = disable_proxy
        self.start_error: str | None = None
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._robots_cache: dict[str, tuple[robotparser.RobotFileParser | None, str | None]] = {}

    def __enter__(self) -> "Screenshotter":
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.content_dir.mkdir(parents=True, exist_ok=True)
        try:
            from playwright.sync_api import sync_playwright  # type: ignore

            self._playwright = sync_playwright().start()
            launch_kwargs: dict[str, Any] = {"headless": True}
            if self.disable_proxy:
                launch_kwargs["args"] = ["--no-proxy-server"]
            self._browser = self._playwright.chromium.launch(**launch_kwargs)
            self._context = self._browser.new_context(
                viewport={"width": self.viewport_width, "height": self.viewport_height},
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                locale="zh-CN",
                ignore_https_errors=True,
            )
        except Exception as exc:
            self.start_error = (
                f"Playwright screenshot unavailable: {safe_error(exc)}. "
                "Install with `pip install playwright` and `python -m playwright install chromium`."
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for resource in [self._context, self._browser]:
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        if self._playwright is not None:
            try:
                self._playwright.stop()
            except Exception:
                pass

    def _scroll_for_lazy_loading(self, page: Any) -> None:
        page.evaluate(
            """
            async () => {
                const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
                const maxSteps = 24;
                let lastHeight = 0;
                let stableRounds = 0;

                for (let step = 0; step < maxSteps; step += 1) {
                    const currentHeight = Math.max(
                        document.body ? document.body.scrollHeight : 0,
                        document.documentElement ? document.documentElement.scrollHeight : 0
                    );

                    window.scrollTo(0, currentHeight);
                    await sleep(700);

                    const nextHeight = Math.max(
                        document.body ? document.body.scrollHeight : 0,
                        document.documentElement ? document.documentElement.scrollHeight : 0
                    );

                    if (nextHeight === lastHeight) {
                        stableRounds += 1;
                    } else {
                        stableRounds = 0;
                    }

                    lastHeight = nextHeight;
                    if (stableRounds >= 2) {
                        break;
                    }
                }

                window.scrollTo(0, 0);
                await sleep(500);
            }
            """
        )

    def _prepare_page(self, page: Any, *, scroll: bool) -> None:
        page.wait_for_timeout(1200)
        try:
            page.wait_for_load_state("networkidle", timeout=8_000)
        except Exception:
            pass
        if scroll:
            try:
                self._scroll_for_lazy_loading(page)
            except Exception:
                pass

    def _extract_page_text(self, page: Any, max_chars: int) -> str:
        text = page.evaluate("() => document.body ? document.body.innerText : ''")
        return clean_page_text(text, max_chars=max_chars)

    def _robots_key(self, url: str) -> str:
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _robots_allowed(self, url: str) -> tuple[bool, str | None]:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return False, "robots skipped: URL is not public http/https."
        key = self._robots_key(url)
        if key not in self._robots_cache:
            parser = robotparser.RobotFileParser()
            parser.set_url(urljoin(key, "/robots.txt"))
            old_timeout = socket.getdefaulttimeout()
            try:
                socket.setdefaulttimeout(5)
                parser.read()
                self._robots_cache[key] = (parser, None)
            except Exception as exc:
                self._robots_cache[key] = (None, f"robots.txt unavailable: {safe_error(exc)}")
            finally:
                socket.setdefaulttimeout(old_timeout)
        parser, warning = self._robots_cache[key]
        if parser is None:
            return True, warning
        try:
            if not parser.can_fetch("*", url):
                return False, "robots.txt disallows fetching this URL."
        except Exception as exc:
            return True, f"robots.txt check failed: {safe_error(exc)}"
        return True, warning

    def _status_from_response(self, response: Any) -> tuple[int | None, str | None, str | None]:
        status = None
        if response is not None:
            try:
                status = int(response.status)
            except Exception:
                status = None
        if status is None:
            return None, None, None
        if status in BLOCK_STATUS_TEXT:
            return status, BLOCK_STATUS_TEXT[status], f"HTTP {status}"
        if status >= 500:
            return status, "error", f"HTTP {status}"
        if status >= 400:
            return status, "blocked", f"HTTP {status}"
        return status, None, None

    def _classify_page_access(
        self,
        page: Any,
        response: Any,
        text: str | None,
        *,
        text_required: bool,
    ) -> tuple[str, int | None, str | None, str | None]:
        http_status, status_from_http, reason_from_http = self._status_from_response(response)
        final_url = None
        try:
            final_url = str(page.url or "") or None
        except Exception:
            final_url = None
        if status_from_http:
            return status_from_http, http_status, final_url, reason_from_http

        cleaned_text = clean_page_text(text or "", max_chars=20_000)
        lowered = cleaned_text.lower()
        diagnostics = self._page_access_diagnostics(page, cleaned_text, final_url)
        for status, patterns in HARD_BLOCK_TEXT_PATTERNS:
            if any(pattern.lower() in lowered for pattern in patterns):
                return status, http_status, final_url, f"Detected {status} marker in page text."
        for status, patterns in SOFT_BLOCK_TEXT_PATTERNS:
            matched = [pattern for pattern in patterns if pattern.lower() in lowered]
            if matched:
                diagnostics["matched_soft_markers"] = matched[:8]
                if not self._soft_marker_means_block(status, diagnostics):
                    continue
                return status, http_status, final_url, f"Detected {status} marker in page text."
        if text_required and not clean_page_text(text or "", max_chars=1000):
            return "empty_dom", http_status, final_url, "No visible body text found after page load."
        return "success", http_status, final_url, None

    def _page_access_diagnostics(self, page: Any, cleaned_text: str, final_url: str | None) -> dict[str, Any]:
        title = ""
        visible_text_chars = len(cleaned_text)
        body_text_chars = visible_text_chars
        article_text_chars = 0
        modal_like = False
        try:
            title = str(page.title() or "")
        except Exception:
            title = ""
        try:
            payload = page.evaluate(
                """
                () => {
                    const textLength = (node) => node && node.innerText ? node.innerText.trim().length : 0;
                    const article = document.querySelector('article, main, [role="main"], .article, .content, .post, .entry-content');
                    const nodes = Array.from(document.querySelectorAll('body *')).slice(0, 600);
                    let modalLike = false;
                    for (const node of nodes) {
                        const style = window.getComputedStyle(node);
                        if (!['fixed', 'sticky'].includes(style.position)) continue;
                        const rect = node.getBoundingClientRect();
                        const area = Math.max(0, rect.width) * Math.max(0, rect.height);
                        const viewportArea = Math.max(1, window.innerWidth * window.innerHeight);
                        const text = (node.innerText || '').toLowerCase();
                        if (area / viewportArea >= 0.25 && /(login|sign in|subscribe|会员|订阅|登录)/.test(text)) {
                            modalLike = true;
                            break;
                        }
                    }
                    return {
                        bodyTextChars: textLength(document.body),
                        articleTextChars: textLength(article),
                        modalLike,
                    };
                }
                """
            )
            if isinstance(payload, dict):
                body_text_chars = int(payload.get("bodyTextChars") or body_text_chars)
                article_text_chars = int(payload.get("articleTextChars") or 0)
                modal_like = bool(payload.get("modalLike"))
        except Exception:
            pass
        return {
            "title": title,
            "url": final_url,
            "visible_text_chars": visible_text_chars,
            "body_text_chars": body_text_chars,
            "article_text_chars": article_text_chars,
            "modal_like": modal_like,
        }

    def _soft_marker_means_block(self, status: str, diagnostics: dict[str, Any]) -> bool:
        text_chars = max(
            int(diagnostics.get("visible_text_chars") or 0),
            int(diagnostics.get("body_text_chars") or 0),
            int(diagnostics.get("article_text_chars") or 0),
        )
        article_chars = int(diagnostics.get("article_text_chars") or 0)
        title = str(diagnostics.get("title") or "").lower()
        url = str(diagnostics.get("url") or "").lower()
        if text_chars >= 1800 or article_chars >= 900:
            return False
        if bool(diagnostics.get("modal_like")):
            return True
        if status == "login_required" and any(hint in url for hint in LOGIN_URL_HINTS):
            return True
        if status == "paywall" and any(hint in title or hint in url for hint in PAYWALL_TITLE_HINTS):
            return True
        if text_chars < 700:
            return True
        return False

    def _extract_dom_text_nodes(self, page: Any) -> list[dict[str, Any]]:
        return page.evaluate(
            """
            () => {
                const normalize = (value) => (value || '').replace(/[ \\t]+/g, ' ').replace(/\\n{3,}/g, '\\n\\n').trim();
                const doc = document.documentElement;
                const body = document.body;
                const pageHeight = Math.max(
                    body ? body.scrollHeight : 0,
                    doc ? doc.scrollHeight : 0,
                    window.innerHeight
                );
                const viewportHeight = window.innerHeight;
                const viewportWidth = window.innerWidth;
                const nodes = [];
                if (!body) {
                    return { nodes, fullText: '', pageHeight, viewportHeight, viewportWidth };
                }

                const hiddenTags = new Set(['script', 'style', 'noscript', 'template', 'svg']);
                const shortValue = (value, maxLength = 240) => value ? String(value).slice(0, maxLength) : '';
                const usableBox = (candidate) => {
                    let current = candidate;
                    let depth = 0;
                    while (current && current !== body && depth < 8) {
                        const candidateStyle = window.getComputedStyle(current);
                        if (
                            candidateStyle.display === 'none' ||
                            candidateStyle.visibility === 'hidden' ||
                            Number(candidateStyle.opacity || '1') === 0
                        ) {
                            return null;
                        }
                        const box = current.getBoundingClientRect();
                        if (box && box.width >= 2 && box.height >= 2) {
                            return box;
                        }
                        current = current.parentElement;
                        depth += 1;
                    }
                    return null;
                };
                const walker = document.createTreeWalker(body, NodeFilter.SHOW_TEXT);
                let node;
                let order = 0;
                while ((node = walker.nextNode())) {
                    const element = node.parentElement;
                    if (!element) continue;
                    const tag = element.tagName.toLowerCase();
                    if (hiddenTags.has(tag)) continue;

                    const style = window.getComputedStyle(element);
                    if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity || '1') === 0) {
                        continue;
                    }

                    const text = normalize(node.nodeValue || '');
                    if (!text || text.length < 2) continue;

                    const range = document.createRange();
                    range.selectNodeContents(node);
                    const rects = Array.from(range.getClientRects()).filter(
                        (rect) => rect.width >= 2 && rect.height >= 2
                    );
                    range.detach();

                    let top = null;
                    let bottom = null;
                    let left = null;
                    let width = null;
                    if (rects.length) {
                        top = Math.min(...rects.map((rect) => rect.top + window.scrollY));
                        bottom = Math.max(...rects.map((rect) => rect.bottom + window.scrollY));
                        left = Math.min(...rects.map((rect) => rect.left + window.scrollX));
                        width = Math.max(...rects.map((rect) => rect.right + window.scrollX)) - left;
                    } else {
                        const box = usableBox(element);
                        if (box && box.width >= 2 && box.height >= 2) {
                            top = box.top + window.scrollY;
                            bottom = box.bottom + window.scrollY;
                            left = box.left + window.scrollX;
                            width = box.width;
                        }
                    }

                    const contextParts = [];
                    let currentElement = element;
                    let depth = 0;
                    while (currentElement && currentElement !== body && depth < 6) {
                        const currentTag = currentElement.tagName ? currentElement.tagName.toLowerCase() : '';
                        const currentClass = typeof currentElement.className === 'string' ? currentElement.className : '';
                        const currentId = currentElement.id || '';
                        const currentRole = currentElement.getAttribute('role') || '';
                        const currentAria = currentElement.getAttribute('aria-label') || '';
                        contextParts.push(
                            [currentTag, currentId, currentClass, currentRole, currentAria]
                                .filter(Boolean)
                                .join(' ')
                        );
                        currentElement = currentElement.parentElement;
                        depth += 1;
                    }

                    nodes.push({
                        order,
                        text,
                        tag,
                        id: shortValue(element.id),
                        className: shortValue(typeof element.className === 'string' ? element.className : ''),
                        role: shortValue(element.getAttribute('role') || ''),
                        ariaLabel: shortValue(element.getAttribute('aria-label') || ''),
                        context: shortValue(contextParts.join(' '), 1200),
                        top,
                        bottom,
                        left,
                        width,
                        length: text.length
                    });
                    order += 1;
                }

                const fullText = nodes.map((item) => item.text).join('\\n');
                let cursor = 0;
                for (const item of nodes) {
                    item.start = cursor;
                    item.end = cursor + item.text.length;
                    cursor = item.end + 1;
                }
                return { nodes, fullText, pageHeight, viewportHeight, viewportWidth };
            }
            """
        )

    def _extract_visible_viewport_text(self, page: Any, max_chars: int) -> str:
        text = page.evaluate(
            """
            () => {
                const normalize = (value) => (value || '').replace(/[ \\t]+/g, ' ').replace(/\\n{3,}/g, '\\n\\n').trim();
                const viewportTop = 0;
                const viewportBottom = window.innerHeight;
                const items = [];
                const seen = new Set();
                if (!document.body) {
                    return '';
                }
                const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                let node;
                let order = 0;
                while ((node = walker.nextNode())) {
                    const element = node.parentElement;
                    if (!element) {
                        continue;
                    }
                    const tag = element.tagName.toLowerCase();
                    if (['script', 'style', 'noscript', 'template', 'svg'].includes(tag)) {
                        continue;
                    }
                    const style = window.getComputedStyle(element);
                    if (style.display === 'none' || style.visibility === 'hidden' || Number(style.opacity || '1') === 0) {
                        continue;
                    }
                    const range = document.createRange();
                    range.selectNodeContents(node);
                    const rects = Array.from(range.getClientRects()).filter(
                        (rect) => rect.width >= 2 && rect.height >= 2
                    );
                    range.detach();
                    const visibleRects = rects.filter(
                        (rect) => rect.bottom >= viewportTop && rect.top <= viewportBottom
                    );
                    if (!visibleRects.length) {
                        continue;
                    }
                    const text = normalize(node.nodeValue || '');
                    if (!text || text.length < 2) {
                        continue;
                    }
                    const firstRect = visibleRects[0];
                    const totalHeight = rects.reduce((sum, rect) => sum + Math.max(0, rect.height), 0);
                    const visibleHeight = visibleRects.reduce((sum, rect) => {
                        const visibleTop = Math.max(rect.top, viewportTop);
                        const visibleBottom = Math.min(rect.bottom, viewportBottom);
                        return sum + Math.max(0, visibleBottom - visibleTop);
                    }, 0);
                    const key = `${Math.round(firstRect.top)}:${Math.round(firstRect.left)}:${text.slice(0, 160)}`;
                    if (seen.has(key)) {
                        continue;
                    }
                    seen.add(key);
                    items.push({
                        top: firstRect.top,
                        left: firstRect.left,
                        order,
                        visibleRatio: visibleHeight / Math.max(1, totalHeight),
                        text
                    });
                    order += 1;
                }
                items.sort((a, b) => (a.top - b.top) || (a.left - b.left) || (a.order - b.order));
                const selectedText = [];
                for (const item of items) {
                    if (selectedText[selectedText.length - 1] === item.text) {
                        continue;
                    }
                    selectedText.push(item.text);
                }
                return selectedText.join('\\n');
            }
            """
        )
        return clean_page_text(text, max_chars=max_chars)

    def _write_page_text(self, url: str, prefix: str, text: str) -> str:
        target = self.content_dir / filename_for_url(prefix, url, extension="txt")
        target.write_text(text, encoding="utf-8")
        return target.relative_to(self.run_dir).as_posix()

    def _write_dom_full_text(self, url: str, prefix: str, text: str) -> str:
        target = self.content_dir / filename_for_url(f"{prefix}_dom_full", url, extension="txt")
        target.write_text(text, encoding="utf-8")
        return target.relative_to(self.run_dir).as_posix()

    def _write_chunk_text(self, url: str, prefix: str, text: str) -> str:
        target = self.content_dir / filename_for_url(prefix, url, extension="txt")
        target.write_text(text, encoding="utf-8")
        return target.relative_to(self.run_dir).as_posix()

    def capture(self, url: str, prefix: str) -> tuple[str | None, str | None]:
        if not is_public_http_url(url):
            return None, "Screenshot skipped: URL is not a public http/https URL."
        if self.start_error:
            return None, self.start_error
        if self._context is None:
            return None, "Screenshot skipped: Playwright context was not initialized."

        target = self.screenshot_dir / filename_for_url(prefix, url)
        try:
            page = self._context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                self._prepare_page(page, scroll=self.full_page)
                page.screenshot(path=str(target), full_page=self.full_page)
            finally:
                page.close()
        except Exception as exc:
            return None, safe_error(exc)

        return target.relative_to(self.run_dir).as_posix(), None

    def process_result_page(
        self,
        url: str,
        prefix: str,
        *,
        capture_screenshot: bool,
        extract_content: bool,
        content_max_chars: int,
    ) -> PageData:
        data = PageData()
        if not is_public_http_url(url):
            message = "Skipped: URL is not a public http/https URL."
            data.access_status = "blocked"
            data.block_reason = message
            if capture_screenshot:
                data.screenshot_error = message
            if extract_content:
                data.content_error = message
            return data
        allowed, robots_warning = self._robots_allowed(url)
        data.robots_warning = robots_warning
        if not allowed:
            data.access_status = "robots_disallowed"
            data.block_reason = robots_warning
            message = robots_warning or "robots.txt disallows fetching this URL."
            if capture_screenshot:
                data.screenshot_error = message
            if extract_content:
                data.content_error = message
            return data
        if self.start_error:
            data.access_status = "error"
            data.block_reason = self.start_error
            if capture_screenshot:
                data.screenshot_error = self.start_error
            if extract_content:
                data.content_error = self.start_error
            return data
        if self._context is None:
            message = "Skipped: Playwright context was not initialized."
            data.access_status = "error"
            data.block_reason = message
            if capture_screenshot:
                data.screenshot_error = message
            if extract_content:
                data.content_error = message
            return data

        last_error: str | None = None
        for attempt in range(2):
            page = self._context.new_page()
            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                self._prepare_page(page, scroll=self.full_page)
                page_text = self._extract_page_text(page, max_chars=max(content_max_chars, 4000))
                status, http_status, final_url, block_reason = self._classify_page_access(
                    page,
                    response,
                    page_text,
                    text_required=extract_content,
                )
                data.access_status = status
                data.http_status = http_status
                data.final_url = final_url
                data.block_reason = block_reason
                data.access_diagnostics = self._page_access_diagnostics(page, page_text, final_url)
                if status not in {"success"}:
                    if extract_content:
                        data.content_error = block_reason or status
                    if capture_screenshot and status in {"blocked", "captcha", "login_required", "paywall", "rate_limited"}:
                        data.screenshot_error = block_reason or status
                    return data

                if extract_content:
                    text = clean_page_text(page_text, max_chars=content_max_chars)
                    if text:
                        data.content_path = self._write_page_text(url, prefix, text)
                        data.content_chars = len(text)
                        data.content_preview = clean_text(text, max_chars=700)
                        data.content_error = None
                        data.content_source = "playwright"
                    else:
                        data.content_error = "No visible body text found after page load."
                        data.access_status = "empty_dom"
                        data.block_reason = data.content_error

                if capture_screenshot:
                    target = self.screenshot_dir / filename_for_url(prefix, url)
                    page.screenshot(path=str(target), full_page=self.full_page)
                    data.screenshot_path = target.relative_to(self.run_dir).as_posix()
                    if data.access_diagnostics is not None:
                        data.access_diagnostics["screenshot_ok"] = True

                return data
            except Exception as exc:
                last_error = safe_error(exc)
                if attempt == 0:
                    try:
                        page.wait_for_timeout(1000)
                    except Exception:
                        pass
            finally:
                try:
                    page.close()
                except Exception:
                    pass

        if capture_screenshot:
            data.screenshot_error = last_error or "Screenshot failed."
        if extract_content and not data.content_path:
            data.content_error = last_error or "Page text extraction failed."
        data.access_status = "timeout" if last_error and "timeout" in last_error.lower() else "error"
        data.block_reason = last_error
        return data

    def capture_evidence_quote(
        self,
        url: str,
        quote: str,
        prefix: str,
        *,
        start_char: int | None = None,
        end_char: int | None = None,
    ) -> tuple[str | None, str | None]:
        if not quote.strip():
            return None, "Evidence quote is empty."
        if not is_public_http_url(url):
            return None, "Evidence screenshot skipped: URL is not a public http/https URL."
        if self.start_error:
            return None, self.start_error
        if self._context is None:
            return None, "Evidence screenshot skipped: Playwright context was not initialized."

        target = self.run_dir / "evidence_shots" / filename_for_url(prefix, url)
        target.parent.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        try:
            page = self._context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                self._prepare_page(page, scroll=True)
                element = self._locate_evidence_element(page, quote)
                if element is not None:
                    page.wait_for_timeout(500)
                    element.screenshot(path=str(target))
                    return target.relative_to(self.run_dir).as_posix(), None
                errors.append("exact/fuzzy quote matching did not find a visible DOM element")

                if start_char is not None:
                    if self._capture_by_text_position(page, start_char, end_char, target):
                        return target.relative_to(self.run_dir).as_posix(), "Used character-position viewport fallback."
                    errors.append("character-position viewport fallback failed")
            finally:
                page.close()
        except Exception as exc:
            return None, safe_error(exc)
        return None, "; ".join(errors) or "Evidence quote was not found in visible DOM text."

    def _locate_evidence_element(self, page: Any, quote: str) -> Any | None:
        candidates = evidence_quote_candidates(quote)
        for candidate in candidates:
            handle = page.evaluate_handle(
                """
                (quote) => {
                    const normalize = (value) => (value || '').replace(/\\s+/g, ' ').trim();
                    const wanted = normalize(quote);
                    if (!wanted) return null;

                    const bestVisibleElement = (node) => {
                        let element = node.parentElement;
                        let best = null;
                        while (element) {
                            const box = element.getBoundingClientRect();
                            const style = window.getComputedStyle(element);
                            const visible = box.width > 20 && box.height > 10 && style.visibility !== 'hidden' && style.display !== 'none';
                            if (visible) {
                                best = element;
                                if (box.width <= 1100 && box.height <= 700) break;
                            }
                            element = element.parentElement;
                        }
                        if (best) {
                            best.scrollIntoView({ block: 'center', inline: 'nearest' });
                        }
                        return best;
                    };

                    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                    let node;
                    while ((node = walker.nextNode())) {
                        const text = normalize(node.nodeValue);
                        if (text && text.includes(wanted)) {
                            return bestVisibleElement(node);
                        }
                    }

                    const all = Array.from(document.body.querySelectorAll('p, li, td, th, div, section, article, span, h1, h2, h3, h4'));
                    for (const element of all) {
                        const text = normalize(element.innerText || element.textContent);
                        if (!text || !text.includes(wanted)) continue;
                        const box = element.getBoundingClientRect();
                        const style = window.getComputedStyle(element);
                        if (box.width > 20 && box.height > 10 && style.visibility !== 'hidden' && style.display !== 'none') {
                            element.scrollIntoView({ block: 'center', inline: 'nearest' });
                            return element;
                        }
                    }

                    return null;
                }
                """,
                candidate,
            )
            element = handle.as_element()
            if element is not None:
                return element
        return None

    def _capture_by_text_position(
        self,
        page: Any,
        start_char: int,
        end_char: int | None,
        target: Path,
    ) -> bool:
        position = page.evaluate(
            """
            ({ startChar, endChar }) => {
                const body = document.body;
                if (!body) return null;
                const text = body.innerText || '';
                const textLength = Math.max(text.length, 1);
                const midpoint = Math.max(0, Math.min(textLength, Math.floor(((startChar || 0) + (endChar || startChar || 0)) / 2)));
                const ratio = midpoint / textLength;
                const doc = document.documentElement;
                const maxScroll = Math.max(0, doc.scrollHeight - window.innerHeight);
                const y = Math.max(0, Math.min(maxScroll, Math.floor(maxScroll * ratio)));
                window.scrollTo(0, y);
                return { y, textLength, ratio };
            }
            """,
            {"startChar": start_char, "endChar": end_char},
        )
        if not position:
            return False
        page.wait_for_timeout(700)
        raw_viewport = page.viewport_size
        viewport = raw_viewport if isinstance(raw_viewport, dict) else {}
        width = max(1, int(viewport.get("width") or 1280))
        height = max(1, int(viewport.get("height") or 720))
        clip_y = min(80, max(0, height - 1))
        clip_height = max(1, min(520, height - clip_y))
        page.screenshot(
            path=str(target),
            full_page=False,
            clip={"x": 0, "y": clip_y, "width": width, "height": clip_height},
        )
        return target.exists()

    def capture_chunk_position(
        self,
        url: str,
        start_char: int,
        end_char: int,
        prefix: str,
    ) -> tuple[str | None, str | None]:
        if not is_public_http_url(url):
            return None, "Chunk screenshot skipped: URL is not a public http/https URL."
        if self.start_error:
            return None, self.start_error
        if self._context is None:
            return None, "Chunk screenshot skipped: Playwright context was not initialized."

        target = self.run_dir / "chunk_shots" / filename_for_url(prefix, url)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            page = self._context.new_page()
            try:
                page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                self._prepare_page(page, scroll=True)
                text_length = page.evaluate("() => document.body ? (document.body.innerText || '').length : 0")
                if not text_length:
                    return None, "Chunk screenshot failed: page innerText is empty."
                if not self._capture_by_text_position(page, start_char, end_char, target):
                    return None, "Chunk screenshot failed: character-position viewport capture failed."
            finally:
                page.close()
        except Exception as exc:
            return None, safe_error(exc)
        return target.relative_to(self.run_dir).as_posix(), None

    def load_quote_dom_page(self, url: str) -> tuple[Any | None, list[dict[str, Any]], str | None]:
        if not is_public_http_url(url):
            return None, [], "URL is not a public http/https URL."
        allowed, robots_warning = self._robots_allowed(url)
        if not allowed:
            return None, [], robots_warning or "robots.txt disallows fetching this URL."
        if self.start_error:
            return None, [], self.start_error
        if self._context is None:
            return None, [], "Playwright context was not initialized."
        try:
            page = self._context.new_page()
            response = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
            self._prepare_page(page, scroll=True)
            page_text = self._extract_page_text(page, max_chars=4000)
            status, http_status, _, block_reason = self._classify_page_access(
                page,
                response,
                page_text,
                text_required=True,
            )
            if status != "success":
                page.close()
                status_text = f"{status}: {block_reason or 'page access failed'}"
                if http_status is not None:
                    status_text = f"{status_text} (HTTP {http_status})"
                return None, [], status_text
            raw_dom_payload = self._extract_dom_text_nodes(page)
            dom_payload = raw_dom_payload if isinstance(raw_dom_payload, dict) else {}
            dom_nodes = [item for item in dom_payload.get("nodes") or [] if dom_node_text(item)]
            if not dom_nodes:
                page.close()
                return None, [], "No visible DOM text nodes found after page load."
            return page, dom_nodes, None
        except Exception as exc:
            try:
                page.close()  # type: ignore[possibly-undefined]
            except Exception:
                pass
            return None, [], safe_error(exc)

    def locate_quote_on_page(
        self,
        page: Any,
        dom_nodes: list[dict[str, Any]],
        url: str,
        quote: str,
        prefix: str,
        *,
        min_score: float,
    ) -> tuple[QuoteMatch | None, str | None, str | None]:
        if not quote.strip():
            return None, None, "Evidence quote is empty."
        if not dom_nodes:
            return None, None, "No visible DOM text nodes found after page load."

        target = self.run_dir / "quote_shots" / filename_for_url(prefix, url)
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            match = find_quote_match(dom_nodes, quote, min_score=min_score)
            if match is None:
                return None, None, "Evidence quote was not found in visible DOM text."

            try:
                shot_path = self._capture_quote_match(page, dom_nodes, match, target)
            except Exception as exc:
                return match, None, safe_error(exc)
            if shot_path is None:
                return match, None, "Quote matched, but screenshot capture failed."
            return match, shot_path, None
        except Exception as exc:
            return None, None, safe_error(exc)

    def locate_quote(
        self,
        url: str,
        quote: str,
        prefix: str,
        *,
        min_score: float,
    ) -> tuple[QuoteMatch | None, str | None, str | None]:
        page, dom_nodes, error = self.load_quote_dom_page(url)
        if error or page is None:
            return None, None, error
        try:
            return self.locate_quote_on_page(page, dom_nodes, url, quote, prefix, min_score=min_score)
        finally:
            try:
                page.close()
            except Exception:
                pass

    def _capture_quote_match(
        self,
        page: Any,
        dom_nodes: list[dict[str, Any]],
        match: QuoteMatch,
        target: Path,
    ) -> str | None:
        matched_nodes = [
            node
            for node in dom_nodes
            if int(node.get("end") or 0) > match.dom_start and int(node.get("start") or 0) < match.dom_end
        ]
        usable_nodes = [
            node
            for node in matched_nodes
            if isinstance(node.get("top"), (int, float)) and isinstance(node.get("bottom"), (int, float))
        ]
        if not usable_nodes:
            usable_nodes = [
                node
                for node in dom_nodes
                if isinstance(node.get("top"), (int, float)) and isinstance(node.get("bottom"), (int, float))
            ]
        if not usable_nodes:
            return None

        top = min(float(node.get("top")) for node in usable_nodes)
        bottom = max(float(node.get("bottom")) for node in usable_nodes)
        match.page_top = int(top)
        match.page_bottom = int(bottom)
        raw_viewport = page.viewport_size
        viewport = raw_viewport if isinstance(raw_viewport, dict) else {}
        viewport_height = max(1, int(viewport.get("height") or self.viewport_height))
        scroll_y = max(0, int(top - (viewport_height * 0.35)))
        page.evaluate("(y) => window.scrollTo(0, y)", scroll_y)
        page.wait_for_timeout(700)
        page.evaluate(
            """
            (ranges) => {
                document.querySelectorAll('[data-quote-locator-highlight="1"]').forEach((node) => node.remove());
                const colors = {
                    border: 'rgba(245, 158, 11, 0.95)',
                    fill: 'rgba(245, 158, 11, 0.18)'
                };
                for (const range of ranges) {
                    const top = Math.max(0, range.top - window.scrollY - 4);
                    const left = Math.max(0, (range.left || 0) - window.scrollX - 4);
                    const width = Math.max(48, Math.min(window.innerWidth - left - 8, range.width || window.innerWidth - 16));
                    const height = Math.max(20, Math.min(window.innerHeight - top - 8, range.height || 32) + 8);
                    const box = document.createElement('div');
                    box.setAttribute('data-quote-locator-highlight', '1');
                    Object.assign(box.style, {
                        position: 'fixed',
                        top: `${top}px`,
                        left: `${left}px`,
                        width: `${width}px`,
                        height: `${height}px`,
                        border: `3px solid ${colors.border}`,
                        background: colors.fill,
                        boxShadow: '0 0 0 99999px rgba(0, 0, 0, 0.03)',
                        pointerEvents: 'none',
                        zIndex: '2147483647',
                        borderRadius: '4px',
                        boxSizing: 'border-box'
                    });
                    document.body.appendChild(box);
                }
            }
            """,
            [
                {
                    "top": float(node.get("top") or top),
                    "left": float(node.get("left") or 0),
                    "width": float(node.get("width") or min(self.viewport_width - 16, 1100)),
                    "height": max(18, float(node.get("bottom") or bottom) - float(node.get("top") or top)),
                }
                for node in usable_nodes[:8]
            ],
        )
        page.screenshot(path=str(target), full_page=False)
        return target.relative_to(self.run_dir).as_posix() if target.exists() else None

    def capture_chunk_range_on_page(
        self,
        page: Any,
        dom_nodes: list[dict[str, Any]],
        url: str,
        chunk_text_value: str,
        prefix: str,
        *,
        range_start: int | None = None,
        range_end: int | None = None,
        min_score: float = 0.72,
        boundary_chars: int = 120,
    ) -> ChunkShotCapture:
        text = clean_page_text(chunk_text_value, max_chars=1_000_000)
        if not text:
            return ChunkShotCapture(paths=[], error="Chunk text is empty.")
        if not dom_nodes:
            return ChunkShotCapture(paths=[], error="No visible DOM text nodes found after page load.")

        start_quote = text[:boundary_chars]
        end_quote = text[-boundary_chars:] if len(text) > boundary_chars else text
        start_match = find_quote_match(dom_nodes, start_quote, min_score=min_score)
        end_match = find_quote_match(dom_nodes, end_quote, min_score=min_score)
        if start_match is None:
            return ChunkShotCapture(paths=[], error="Chunk start text was not found in visible DOM text.")
        if end_match is None:
            return ChunkShotCapture(paths=[], error="Chunk end text was not found in visible DOM text.")
        if start_match.page_top is None or end_match.page_bottom is None:
            if range_start is None or range_end is None:
                return ChunkShotCapture(paths=[], error="Chunk boundary text matched, but page coordinates were unavailable.")
            fallback_top, fallback_bottom = text_range_page_bounds(dom_nodes, range_start, range_end)
            if fallback_top is None or fallback_bottom is None:
                return ChunkShotCapture(paths=[], error="Chunk boundary text matched, but page coordinates were unavailable.")
            return self._capture_page_vertical_range(page, url, prefix, max(0, fallback_top - 24), fallback_bottom + 24)

        top = max(0, min(int(start_match.page_top), int(end_match.page_top or start_match.page_top)) - 24)
        bottom = max(int(start_match.page_bottom or top), int(end_match.page_bottom)) + 24
        return self._capture_page_vertical_range(page, url, prefix, top, bottom)

    def _capture_page_vertical_range(
        self,
        page: Any,
        url: str,
        prefix: str,
        page_top: int,
        page_bottom: int,
        segment_overlap_px: int = 120,
    ) -> ChunkShotCapture:
        target_dir = self.run_dir / "chunk_shots"
        target_dir.mkdir(parents=True, exist_ok=True)
        raw_viewport = page.viewport_size
        viewport = raw_viewport if isinstance(raw_viewport, dict) else {}
        width = max(1, int(viewport.get("width") or self.viewport_width))
        viewport_height = max(1, int(viewport.get("height") or self.viewport_height))
        page_height = int(
            page.evaluate(
                """
                () => Math.max(
                    document.body ? document.body.scrollHeight : 0,
                    document.documentElement ? document.documentElement.scrollHeight : 0,
                    window.innerHeight
                )
                """
            )
            or viewport_height
        )
        top = max(0, min(page_top, max(0, page_height - 1)))
        bottom = max(top + 1, min(page_bottom, page_height))
        max_scroll = max(0, page_height - viewport_height)
        paths: list[str] = []
        shot_index = 1
        current = top
        overlap_px = max(0, min(int(segment_overlap_px), max(0, viewport_height - 1)))
        while current < bottom:
            scroll_y = min(current, max_scroll)
            page.evaluate("(y) => window.scrollTo(0, y)", scroll_y)
            page.wait_for_timeout(350)
            clip_y = max(0, current - scroll_y)
            available = max(1, viewport_height - clip_y)
            clip_height = max(1, min(available, bottom - current))
            target = target_dir / filename_for_url(f"{prefix}_part_{shot_index:03d}", url)
            page.screenshot(
                path=str(target),
                full_page=False,
                clip={"x": 0, "y": int(clip_y), "width": width, "height": int(clip_height)},
            )
            if target.exists():
                paths.append(target.relative_to(self.run_dir).as_posix())
            covered_end = current + clip_height
            if covered_end >= bottom:
                break
            effective_overlap = min(overlap_px, max(0, clip_height - 1))
            current = max(current + 1, covered_end - effective_overlap)
            shot_index += 1
            if shot_index > 200:
                return ChunkShotCapture(
                    paths=paths,
                    error="Chunk screenshot stopped after 200 parts.",
                    scroll_y=top,
                    page_top=top,
                    page_bottom=bottom,
                )
        return ChunkShotCapture(paths=paths, scroll_y=top, page_top=top, page_bottom=bottom)

    def capture_viewport_chunks(
        self,
        url: str,
        prefix: str,
        *,
        max_scrolls: int,
        overlap_ratio: float,
        text_max_chars: int,
        text_overlap_chars: int,
        capture_screenshots: bool,
        max_screenshots: int | None = None,
    ) -> tuple[list[ViewportChunkData], str | None]:
        if not is_public_http_url(url):
            return [], "Viewport chunking skipped: URL is not a public http/https URL."
        allowed, robots_warning = self._robots_allowed(url)
        if not allowed:
            return [], robots_warning or "robots.txt disallows fetching this URL."
        if self.start_error:
            return [], self.start_error
        if self._context is None:
            return [], "Viewport chunking skipped: Playwright context was not initialized."

        overlap_ratio = max(0.0, min(0.95, overlap_ratio))
        target_dir = self.run_dir / "chunk_shots"
        if capture_screenshots:
            target_dir.mkdir(parents=True, exist_ok=True)

        try:
            page = self._context.new_page()
            try:
                response = page.goto(url, wait_until="domcontentloaded", timeout=45_000)
                self._prepare_page(page, scroll=True)
                page_text = self._extract_page_text(page, max_chars=4000)
                status, http_status, _, block_reason = self._classify_page_access(
                    page,
                    response,
                    page_text,
                    text_required=True,
                )
                if status != "success":
                    status_text = f"{status}: {block_reason or 'page access failed'}"
                    if http_status is not None:
                        status_text = f"{status_text} (HTTP {http_status})"
                    return [], status_text
                raw_dom_payload = self._extract_dom_text_nodes(page)
                dom_payload = raw_dom_payload if isinstance(raw_dom_payload, dict) else {}
                dom_nodes = [item for item in dom_payload.get("nodes") or [] if clean_text(item.get("text") or "", max_chars=10_000)]
                full_dom_text = clean_page_text(dom_payload.get("fullText") or "", max_chars=1_000_000)
                full_dom_text_path = self._write_dom_full_text(url, prefix, full_dom_text) if full_dom_text else None
                page_height = int(dom_payload.get("pageHeight") or 0)
                viewport_height = int(dom_payload.get("viewportHeight") or 720)
                viewport_width = int(dom_payload.get("viewportWidth") or 1280)
                if page_height <= 0 or viewport_height <= 0:
                    return [], "Viewport chunking failed: invalid page or viewport height."

                max_y = max(0, page_height - viewport_height)
                step = max(1, int(viewport_height * (1 - overlap_ratio)))
                scroll_positions: list[int] = []
                scroll_y = 0
                while True:
                    scroll_positions.append(min(scroll_y, max_y))
                    if scroll_y >= max_y:
                        break
                    if max_scrolls > 0 and len(scroll_positions) >= max_scrolls + 1:
                        break
                    scroll_y += step
                    if scroll_y >= max_y:
                        if scroll_positions[-1] != max_y:
                            scroll_positions.append(max_y)
                        break

                if not scroll_positions:
                    scroll_positions = [0]

                chunk_node_indexes: list[list[int]] = [[] for _ in scroll_positions]
                assigned_indexes: set[int] = set()
                for node_index, node in enumerate(dom_nodes):
                    top = node.get("top")
                    bottom = node.get("bottom")
                    if not isinstance(top, (int, float)) or not isinstance(bottom, (int, float)):
                        continue
                    best_index: int | None = None
                    best_overlap = 0.0
                    for viewport_index, position in enumerate(scroll_positions):
                        view_top = float(position)
                        view_bottom = float(position + viewport_height)
                        overlap = max(0.0, min(float(bottom), view_bottom) - max(float(top), view_top))
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_index = viewport_index
                    if best_index is not None and best_overlap > 0:
                        chunk_node_indexes[best_index].append(node_index)
                        assigned_indexes.add(node_index)

                gap_filled_counts = [0 for _ in scroll_positions]
                for node_index, node in enumerate(dom_nodes):
                    if node_index in assigned_indexes:
                        continue
                    top = node.get("top")
                    if isinstance(top, (int, float)):
                        best_index = min(
                            range(len(scroll_positions)),
                            key=lambda index: abs(float(top) - float(scroll_positions[index])),
                        )
                    else:
                        ratio = node_index / max(1, len(dom_nodes) - 1)
                        best_index = min(len(scroll_positions) - 1, max(0, round(ratio * (len(scroll_positions) - 1))))
                    chunk_node_indexes[best_index].append(node_index)
                    gap_filled_counts[best_index] += 1

                for indexes in chunk_node_indexes:
                    indexes.sort()

                chunks: list[ViewportChunkData] = []
                seen_positions: set[int] = set()
                for viewport_index, position in enumerate(scroll_positions, start=1):
                    if position in seen_positions:
                        continue
                    seen_positions.add(position)
                    base_node_indexes = chunk_node_indexes[viewport_index - 1]
                    if base_node_indexes and text_overlap_chars > 0:
                        first_index = base_node_indexes[0]
                        last_index = base_node_indexes[-1]
                        prefix_indexes: list[int] = []
                        suffix_indexes: list[int] = []
                        prefix_chars = 0
                        for candidate in range(first_index - 1, -1, -1):
                            prefix_indexes.insert(0, candidate)
                            prefix_chars += len(str(dom_nodes[candidate].get("text") or "")) + 1
                            if prefix_chars >= text_overlap_chars:
                                break
                        suffix_chars = 0
                        for candidate in range(last_index + 1, len(dom_nodes)):
                            suffix_indexes.append(candidate)
                            suffix_chars += len(str(dom_nodes[candidate].get("text") or "")) + 1
                            if suffix_chars >= text_overlap_chars:
                                break
                        base_node_indexes = sorted(set(prefix_indexes + base_node_indexes + suffix_indexes))

                    node_index_groups = split_node_indexes_by_text_length(
                        dom_nodes,
                        base_node_indexes,
                        max_chars=text_max_chars,
                        overlap_chars=text_overlap_chars,
                    )
                    if not node_index_groups:
                        node_index_groups = [[]]

                    page.evaluate("(y) => window.scrollTo(0, y)", position)
                    page.wait_for_timeout(700)
                    shared_shot_path: str | None = None
                    shared_shot_error: str | None = None
                    for sub_index, node_indexes in enumerate(node_index_groups, start=1):
                        text_parts = [dom_node_text(dom_nodes[node_index]) for node_index in node_indexes]
                        text = clean_page_text("\n".join(part for part in text_parts if part), max_chars=text_max_chars)
                        if not text and full_dom_text and len(scroll_positions) == 1:
                            text = clean_page_text(full_dom_text, max_chars=text_max_chars)

                        shot_path: str | None = shared_shot_path
                        shot_error: str | None = shared_shot_error
                        if sub_index == 1:
                            if capture_screenshots and text and (max_screenshots is None or viewport_index <= max_screenshots):
                                target = target_dir / filename_for_url(f"{prefix}_viewport_{viewport_index:04d}", url)
                                try:
                                    page.screenshot(path=str(target), full_page=False)
                                    shot_path = target.relative_to(self.run_dir).as_posix()
                                except Exception as exc:
                                    shot_error = safe_error(exc)
                            elif capture_screenshots and text:
                                shot_error = "Viewport chunk screenshot skipped: max evidence shot limit reached."
                            shared_shot_path = shot_path
                            shared_shot_error = shot_error

                        dom_range_start: int | None = None
                        dom_range_end: int | None = None
                        if node_indexes:
                            starts = [int(dom_nodes[node_index].get("start") or 0) for node_index in node_indexes]
                            ends = [int(dom_nodes[node_index].get("end") or 0) for node_index in node_indexes]
                            dom_range_start = min(starts) if starts else None
                            dom_range_end = max(ends) if ends else None
                        mismatch_warning = None
                        if node_indexes:
                            view_bottom = position + viewport_height
                            outside_count = 0
                            for node_index in node_indexes:
                                node = dom_nodes[node_index]
                                top = node.get("top")
                                bottom = node.get("bottom")
                                if isinstance(top, (int, float)) and isinstance(bottom, (int, float)):
                                    if max(0.0, min(float(bottom), float(view_bottom)) - max(float(top), float(position))) <= 0:
                                        outside_count += 1
                            if outside_count:
                                mismatch_warning = (
                                    f"{outside_count} DOM text segment(s) were assigned to this chunk "
                                    "but may not be visible in the screenshot because viewport coverage was limited."
                                )
                        if len(node_index_groups) > 1:
                            extra = f"DOM text for this viewport was split into part {sub_index}/{len(node_index_groups)}."
                            mismatch_warning = f"{mismatch_warning} {extra}".strip() if mismatch_warning else extra
                        chunks.append(
                            ViewportChunkData(
                                text=text,
                                start_char=dom_range_start or 0,
                                end_char=dom_range_end or len(text),
                                scroll_y=position,
                                viewport_height=viewport_height,
                                viewport_width=viewport_width,
                                page_height=page_height,
                                dom_text_length=len(full_dom_text),
                                dom_range_start=dom_range_start,
                                dom_range_end=dom_range_end,
                                dom_gap_filled_count=gap_filled_counts[viewport_index - 1],
                                text_sources=["dom"],
                                full_dom_text_path=full_dom_text_path,
                                screenshot_mismatch_warning=mismatch_warning,
                                chunk_shot_path=shot_path,
                                chunk_shot_error=shot_error,
                            )
                        )
                return chunks, None
            finally:
                page.close()
        except Exception as exc:
            return [], safe_error(exc)

def node_indexes_for_text_range(dom_nodes: list[dict[str, Any]], start: int, end: int) -> list[int]:
    return [
        index
        for index, node in enumerate(dom_nodes)
        if int(node.get("end") or 0) > start and int(node.get("start") or 0) < end
    ]


def text_range_page_bounds(dom_nodes: list[dict[str, Any]], start: int, end: int) -> tuple[int | None, int | None]:
    indexes = node_indexes_for_text_range(dom_nodes, start, end)
    nodes = [dom_nodes[index] for index in indexes]
    tops = [node.get("top") for node in nodes if isinstance(node.get("top"), (int, float))]
    bottoms = [node.get("bottom") for node in nodes if isinstance(node.get("bottom"), (int, float))]
    return (int(min(tops)) if tops else None, int(max(bottoms)) if bottoms else None)

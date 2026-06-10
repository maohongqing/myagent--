from myagent.search.browser import Screenshotter


class FakePage:
    def __init__(self, url: str = "https://example.com", title: str = "", article_chars: int = 0, modal_like: bool = False) -> None:
        self.url = url
        self._title = title
        self._article_chars = article_chars
        self._modal_like = modal_like

    def title(self) -> str:
        return self._title

    def evaluate(self, _script: str):
        return {
            "bodyTextChars": self._article_chars,
            "articleTextChars": self._article_chars,
            "modalLike": self._modal_like,
        }


class FakeResponse:
    def __init__(self, status: int) -> None:
        self.status = status


def test_classify_page_access_by_http_status(tmp_path) -> None:
    screenshotter = Screenshotter(tmp_path, full_page=False)

    status, http_status, final_url, reason = screenshotter._classify_page_access(
        FakePage(),
        FakeResponse(429),
        "Rate limited",
        text_required=True,
    )

    assert status == "rate_limited"
    assert http_status == 429
    assert final_url == "https://example.com"
    assert reason == "HTTP 429"


def test_classify_page_access_by_text_markers(tmp_path) -> None:
    screenshotter = Screenshotter(tmp_path, full_page=False)

    captcha = screenshotter._classify_page_access(FakePage(), FakeResponse(200), "Please solve CAPTCHA", text_required=True)
    login = screenshotter._classify_page_access(FakePage(), FakeResponse(200), "请先登录后继续", text_required=True)
    paywall = screenshotter._classify_page_access(FakePage(), FakeResponse(200), "会员专享内容", text_required=True)

    assert captcha[0] == "captcha"
    assert login[0] == "login_required"
    assert paywall[0] == "paywall"


def test_classify_page_access_ignores_soft_markers_in_long_content(tmp_path) -> None:
    screenshotter = Screenshotter(tmp_path, full_page=False)

    body = "Log in Register Subscribe " + ("Useful article content. " * 120)

    status, http_status, final_url, reason = screenshotter._classify_page_access(
        FakePage(),
        FakeResponse(200),
        body,
        text_required=True,
    )

    assert status == "success"
    assert http_status == 200
    assert final_url == "https://example.com"
    assert reason is None


def test_classify_page_access_ignores_login_marker_with_article_body(tmp_path) -> None:
    screenshotter = Screenshotter(tmp_path, full_page=False)
    body = "登录 会员 订阅 " + ("这是一段正常文章正文，介绍产品和行业变化。" * 90)

    status, _, _, reason = screenshotter._classify_page_access(
        FakePage(article_chars=1800),
        FakeResponse(200),
        body,
        text_required=True,
    )

    assert status == "success"
    assert reason is None


def test_classify_page_access_blocks_short_login_page(tmp_path) -> None:
    screenshotter = Screenshotter(tmp_path, full_page=False)

    status, _, _, _ = screenshotter._classify_page_access(
        FakePage(url="https://example.com/login", title="Login"),
        FakeResponse(200),
        "请先登录后继续",
        text_required=True,
    )

    assert status == "login_required"


def test_classify_page_access_empty_dom(tmp_path) -> None:
    screenshotter = Screenshotter(tmp_path, full_page=False)

    status, _, _, reason = screenshotter._classify_page_access(FakePage(), FakeResponse(200), "", text_required=True)

    assert status == "empty_dom"
    assert reason == "No visible body text found after page load."

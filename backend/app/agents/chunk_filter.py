from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from myagent.backend.app.models import RawSource


AD_NAV_TERMS = [
    "advertisement",
    "sponsored",
    "cookie",
    "privacy policy",
    "terms of use",
    "sign in",
    "log in",
    "subscribe",
    "newsletter",
    "related articles",
    "recommended",
    "share this",
    "广告",
    "推广",
    "赞助",
    "登录",
    "注册",
    "隐私政策",
    "用户协议",
    "相关推荐",
    "热门推荐",
    "相关阅读",
    "分享到",
    "上一篇",
    "下一篇",
]

FIELD_TERMS: dict[str, list[str]] = {
    "website": ["official", "website", "官网", "首页"],
    "positioning": ["position", "about", "定位", "简介", "使命"],
    "target_users": ["user", "customer", "audience", "用户", "客户", "人群"],
    "core_scenarios": ["scenario", "use case", "场景", "用途"],
    "core_features": ["feature", "function", "capability", "功能", "能力", "特性"],
    "key_parameters": ["parameter", "spec", "limit", "参数", "规格", "配置"],
    "pricing": ["pricing", "price", "subscription", "plan", "定价", "价格", "收费", "订阅"],
    "business_model": ["business model", "revenue", "monetization", "商业模式", "收入", "营收"],
    "channels": ["channel", "app store", "渠道", "分发", "官网"],
    "growth_signals": ["growth", "mau", "download", "revenue", "增长", "下滑", "用户数", "下载", "营收", "财报"],
    "funding_or_org_signals": ["funding", "financing", "hiring", "融资", "投资", "招聘", "团队"],
    "technology_or_product_features": ["technology", "ai", "api", "技术", "模型", "算法"],
    "differentiation": ["differentiation", "advantage", "特色", "优势", "差异"],
    "user_feedback": ["review", "rating", "feedback", "complaint", "评价", "评论", "投诉", "售后"],
    "risks": ["risk", "complaint", "regulation", "lawsuit", "风险", "投诉", "监管", "争议"],
    "recent_updates": ["news", "release", "update", "launch", "新闻", "发布", "更新"],
    "market_situation.market_size": ["market size", "tam", "市场规模", "规模"],
    "market_situation.growth_trend": ["growth", "cagr", "trend", "增长", "趋势"],
    "competition_situation.competition_intensity": ["competition", "competitor", "竞争", "竞品"],
    "external_environment.regulation_or_laws": ["regulation", "law", "policy", "监管", "法律", "政策"],
    "external_environment.technology_trends": ["technology", "ai", "trend", "技术", "趋势"],
}


@dataclass
class ChunkFilterStats:
    original_count: int = 0
    kept_count: int = 0
    skipped_reasons: Counter[str] = field(default_factory=Counter)

    @property
    def llm_call_count(self) -> int:
        return self.kept_count

    def to_dict(self) -> dict[str, object]:
        return {
            "original_count": self.original_count,
            "kept_count": self.kept_count,
            "skipped_count": self.original_count - self.kept_count,
            "skipped_reasons": dict(self.skipped_reasons),
            "llm_call_count": self.llm_call_count,
        }


def filter_relevant_sources(
    sources: list[RawSource],
    *,
    object_name: str = "",
    context_terms: list[str] | None = None,
    expected_fields: list[str] | None = None,
    min_chars: int = 120,
) -> tuple[list[RawSource], ChunkFilterStats]:
    stats = ChunkFilterStats(original_count=len(sources))
    kept: list[RawSource] = []
    for source in sources:
        reason = chunk_skip_reason(
            source,
            object_name=object_name,
            context_terms=context_terms or [],
            expected_fields=expected_fields or [],
            min_chars=min_chars,
        )
        if reason:
            stats.skipped_reasons[reason] += 1
            continue
        kept.append(source)
    stats.kept_count = len(kept)
    return kept, stats


def chunk_skip_reason(
    source: RawSource,
    *,
    object_name: str = "",
    context_terms: list[str],
    expected_fields: list[str],
    min_chars: int,
) -> str | None:
    content = re.sub(r"\s+", " ", source.content or "").strip()
    if not content:
        return "empty"
    haystack = f"{source.title} {source.url} {content}".lower()
    signals = _terms_for_expected_fields(expected_fields)
    terms = [object_name, *context_terms, *signals]
    terms = [term.strip().lower() for term in terms if term and len(term.strip()) >= 2]
    if len(content) < min_chars and not any(term in haystack for term in terms):
        return "too_short"
    ad_hits = sum(1 for term in AD_NAV_TERMS if term.lower() in haystack)
    if ad_hits >= 4 and len(content) < 1200:
        return "ad_or_navigation"
    if _nav_density(content) > 0.38:
        return "navigation_dense"

    if terms and not any(term in haystack for term in terms):
        return "weak_relevance"
    return None


def _terms_for_expected_fields(fields: list[str]) -> list[str]:
    terms: list[str] = []
    for field_name in fields:
        terms.extend(FIELD_TERMS.get(field_name, []))
        short = field_name.split(".")[-1].replace("_", " ")
        if short:
            terms.append(short)
    return terms


def _nav_density(text: str) -> float:
    lines = [line.strip() for line in re.split(r"[\r\n]+", text) if line.strip()]
    if len(lines) < 8:
        return 0.0
    nav_like = 0
    for line in lines:
        if len(line) <= 28 or re.search(r"^(首页|菜单|登录|注册|关于|产品|价格|博客|新闻|下载|联系我们)$", line, re.I):
            nav_like += 1
    return nav_like / max(1, len(lines))

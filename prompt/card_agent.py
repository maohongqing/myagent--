detail_card_query_system_prompt = """
你负责为竞品资料卡和行业资料卡生成网页搜索 query。
只返回 JSON，不要输出 Markdown、解释文字或额外字段。

输入会包含：
- analysis_goal：本次分析目标。
- market：市场范围。
- product_category：产品品类。
- competitors：竞品名称列表。
- competitor_card_fields：竞品资料卡需要补齐的字段列表。
- industry_card_fields：行业资料卡需要补齐的字段列表。

输出要求：
- competitor_query_templates 必须严格生成 3 条，不多不少。
- industry_queries 必须严格生成 3 条，不多不少。
- 每条竞品 query template 必须包含字面量占位符 {competitor}，不得写入任何具体竞品名称。
- 行业 query 不得包含任何具体竞品名称，除非该名称本身就是行业/品类名。
- query 要面向公开网页证据，优先覆盖官网、帮助文档、价格页、应用商店、用户评价、新闻、行业报告、政策或监管来源。
- target_fields 只能使用输入字段列表中的字段名。

输出 JSON 示例：
{
  "competitor_query_templates": [
    {
      "template": "{competitor} 官网 定价 功能 目标用户",
      "target_fields": ["pricing", "core_features", "target_users"],
      "target_site_types": ["official_site", "pricing_page"],
      "priority": 1,
      "rationale": "用于补齐竞品的定位、功能、定价和用户信息"
    }
  ],
  "industry_queries": [
    {
      "query": "某品类 市场规模 增长趋势 行业报告",
      "target_fields": ["market_situation.market_size", "market_situation.growth_trend"],
      "target_site_types": ["industry_report", "news"],
      "priority": 1,
      "rationale": "用于补齐行业规模和增长趋势"
    }
  ],
  "rationale": "整体查询计划理由"
}
"""

competitor_card_patch_system_prompt = """
你负责根据单个网页 chunk 填充某一个竞品资料卡。
只返回 JSON，不要输出 Markdown、解释文字或额外字段。

输入会包含：
- competitor：当前竞品名称。
- fields_to_fill：本轮仍需要关注的竞品资料卡字段。
- current_card_values：当前已经填写的资料卡 value 快照，只包含字段名和值，不包含 confidence 和 source_refs。
- chunk：网页 chunk 文本与来源信息。

规则：
- 只提取当前 chunk 直接支持的信息，不能凭常识、行业经验或模型记忆补充。
- 竞品名称必须与输入 competitor 保持一致；如果 chunk 明显不是该竞品相关内容，不要填字段。
- 已有字段可以在证据更具体、更完整时补充或替换，但不要重复堆砌同义内容。
- 每个非空字段必须包含 value、confidence、source_refs。
- source_refs 至少要包含 quote；如输入中有 source_id、chunk_id、url、title，也要保留。
- quote 必须从 chunk 原文逐字摘录，不能改写、翻译或编造。
- 如果当前 chunk 不支持任何字段，返回空 fields。
- missing_fields 返回处理完该 chunk 后仍然缺少或证据不足、建议后续 chunk 继续关注的字段。

输出 JSON 示例：
{
  "fields": {
    "pricing": {
      "value": "从 chunk 提取的定价信息",
      "confidence": 0.82,
      "source_refs": [
        {
          "source_id": "raw_123",
          "chunk_id": "chunk_0001",
          "url": "https://example.com/pricing",
          "title": "来源标题",
          "quote": "从原文逐字摘录的证据",
          "screenshot_path": null,
          "evidence_screenshot_path": null,
          "chunk_shot_paths": []
        }
      ]
    }
  },
  "missing_fields": ["growth_signals", "risks"]
}
"""

industry_card_patch_system_prompt = """
你负责根据单个网页 chunk 填充行业资料卡。
只返回 JSON，不要输出 Markdown、解释文字或额外字段。

输入会包含：
- industry：当前行业、市场或产品品类名称。
- fields_to_fill：本轮仍需要关注的行业资料卡字段。
- current_card_values：当前已经填写的资料卡 value 快照，只包含字段名和值，不包含 confidence 和 source_refs。
- chunk：网页 chunk 文本与来源信息。

规则：
- 只提取当前 chunk 直接支持的信息，不能凭常识、行业经验或模型记忆补充。
- 行业信息必须服务于输入 industry 的范围；如果 chunk 与该行业无关，不要填字段。
- 已有字段可以在证据更具体、更完整时补充或替换，但不要重复堆砌同义内容。
- 每个非空字段必须包含 value、confidence、source_refs。
- source_refs 至少要包含 quote；如输入中有 source_id、chunk_id、url、title，也要保留。
- quote 必须从 chunk 原文逐字摘录，不能改写、翻译或编造。
- 如果当前 chunk 不支持任何字段，返回空 fields。
- missing_fields 返回处理完该 chunk 后仍然缺少或证据不足、建议后续 chunk 继续关注的字段。

输出 JSON 示例：
{
  "fields": {
    "market_situation.market_size": {
      "value": "从 chunk 提取的市场规模信息",
      "confidence": 0.82,
      "source_refs": [
        {
          "source_id": "raw_123",
          "chunk_id": "chunk_0001",
          "url": "https://example.com/report",
          "title": "来源标题",
          "quote": "从原文逐字摘录的证据",
          "screenshot_path": null,
          "evidence_screenshot_path": null,
          "chunk_shot_paths": []
        }
      ]
    }
  },
  "missing_fields": ["external_environment.regulation_or_laws"]
}
"""

card_gap_query_system_prompt = """
你负责为未通过质检的资料卡字段生成补充网页搜索 query。
只返回 JSON，不要输出 Markdown、解释文字或额外字段。

输入会包含：
- target_product：用户要分析的目标产品。
- object_type：competitor 或 industry。
- object_name：当前竞品名称或行业名称。
- missing_fields：缺失或证据不足的字段名称。

输出要求：
- queries 最多 3 条。
- 每条 query 必须具体、可直接用于搜索引擎，并以寻找公开证据为目标。
- object_type 为 competitor 时，query 必须包含 object_name。
- object_type 为 industry 时，query 应聚焦行业/品类与缺失字段，不要加入具体竞品名。

输出 JSON 示例：
{
  "queries": [
    {
      "query": "某竞品 定价 商业模式 官网",
      "tool": null,
      "competitor": "某竞品",
      "target_site_types": ["official_site", "pricing_page"],
      "priority": 1,
      "expected_fields": ["pricing", "business_model"],
      "rationale": "补充定价和商业模式的直接证据"
    }
  ]
}
"""

card_qa_system_prompt = """
你是竞品资料卡和行业资料卡的严格质检专家。
只返回 JSON，不要输出 Markdown、解释文字或额外字段。

输入会包含：
- object_type：competitor 或 industry。
- object_name：当前竞品名称或行业名称。
- card：完整资料卡。每个字段只用于检查 value 和 source_refs 中的 quote。

质检标准：
- 字段 value 为空、过泛、明显无信息量，视为缺失。
- 字段 value 不为空但没有 source_refs.quote 支撑，视为未通过。
- quote 必须能直接支撑 value 的关键事实；只出现相同关键词但无法证明结论，视为证据不足。
- 不检查或依赖 confidence、url、截图路径等字段。
- 如果存在关键字段缺失、证据不足或对象名称不一致，passed=false 且 needs_retry=true。
- missing_fields 写仍缺失或需要补搜的字段。
- error_fields 写有值但证据不足、对象不一致或明显错误的字段。
- reference_issues 用简短中文说明证据问题。

输出 JSON 示例：
{
  "passed": false,
  "error_fields": ["growth_signals"],
  "missing_fields": ["risks"],
  "reference_issues": [
    "growth_signals 的 value 声称增长明显，但 quote 只说明产品功能，不能支撑增长判断"
  ],
  "needs_retry": true,
  "reason": "仍有字段缺失或证据不足，需要补充搜索"
}
"""

competitor_card_patch_system_prompt += """

补充硬规则：
- value 的每个关键事实必须能被同一个或多个 quote 直接支撑。
- 不要把公司整体财报、整体 MAU、整体营收写成某个子业务或具体产品的专属指标，除非 quote 明确指向该子业务/产品。
- growth_signals 必须保留时间点和方向；允许写“收入下滑”“增长承压”，不要强行正向总结。
- 单个 chunk 支撑不了完整结论时，只提取窄事实，不要概括成全局判断。
"""

card_qa_system_prompt += """

补充 QA 口径：
- 输入会提供 hard_required_fields 和 optional_fields。只有 hard_required_fields 缺失或证据错误才应 needs_retry=true。
- optional_fields 缺失或证据较弱时，不要设置 needs_retry=true；可写入 reference_issues 或 reason 作为风险提示。
- website 字段允许 source_refs.url 作为证据，不要求 quote 中逐字出现 URL。
- 增长、收入、用户量等指标必须检查时间点、对象范围和方向，不能把公司整体指标误判为具体子业务指标。
"""

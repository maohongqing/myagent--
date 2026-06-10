# Prompt 使用流程与清理说明

本文档按当前后端实际主流程整理 `myagent/prompt` 下 prompt 的调用情况。代码入口主要是：

- `myagent/backend/app/agents/workflow.py`
- `myagent/backend/app/agents/prompt_bridge.py`
- `myagent/backend/app/agents/candidate_selector.py`
- `myagent/backend/app/agents/card_collector.py`
- `myagent/backend/app/agents/tool_executor.py`
- `myagent/backend/app/agents/survey.py`

## 当前保留的 prompt 文件

- `plan_agent.py`
- `search_agent.py`
- `analyst_agent.py`
- `qa_agent.py`
- `card_agent.py`
- `tool_executor_agent.py`
- `survey_agent.py`
- `writer_agent.py`

## 已删除的 prompt 文件

- `structurer_agent.py`

删除原因：

- 当前 `AgentWorkflow.run(...)` 主循环不会进入 `structurer_agent`。
- `_build_graph()` 没有注册 `structurer_agent` 节点。
- `route_after_qa(...)` 不会返回 `structurer_agent`。
- 结构化整理 prompt 已不再参与当前实际执行链路。

配套清理：

- 移除 `prompt_bridge.py` 中的 `STRUCTURER_PROMPT` 和 `structured_knowledge_user_prompt` 导入/导出。
- 移除 `workflow.py` 中未路由调用的 `structurer_agent(...)` 遗留方法。
- 移除前端和模型枚举中的 `structurer_agent` 展示/目标项。
- 删除对应的旧测试 `test_agent_structurer.py`。

## 实际主流程

### 1. Phase 1 规划：解析用户需求

Agent：`plan_agent`

使用 prompt：

- System：`plan_system_prompt`
- User：`phase_1_plan_intent`

调用位置：

- `workflow.py` 的 `plan_agent(...)`

输出：

- `StateTreePatch`
- 更新 `state_tree.phase_1_intent`
- 生成或补齐 `analysis_intent`

补充调用：

- System：`analysis_intent_system_prompt`
- User instruction：`analysis_intent_user_instruction`
- 输出 schema：`AnalysisIntentJSON`

### 2. Phase 2 搜索：生成候选检索 query

Agent：`search_agent`

使用 prompt：

- System：`candidate_query_generation_prompt`

调用位置：

- `workflow.py` 的 `search_agent(...)`

输出：

- `CandidateSearchQueryOutput`
- 后续用于收集候选竞品来源。

### 3. Phase 2 搜索：网页 chunk 抽取候选竞品

Agent：`search_agent`

使用 prompt：

- System：`candidate_chunk_generation_prompt`

调用位置：

- `candidate_selector.py`
- 由 `workflow.py` 调用 `generate_candidate_pool_from_chunks_with_llm_or_rules(...)`

输出：

- `LLMCandidateChunkOutput`
- 后端规则负责候选合并、去重和兜底。

说明：

- 当前没有单独的候选池全量汇总 prompt。

### 4. Phase 2 QA：验证候选真实存在

Agent：`qa_agent`

使用 prompt：

- System：`candidate_existence_qa_system_prompt`
- User instruction：`candidate_existence_qa_instruction`

调用位置：

- `workflow.py` 的候选 QA 流程

输出：

- `CandidateExistenceQAOutput`
- 校验存在性、相关性、规范名称、合并目标和竞品类型。

### 5. Phase 2 分析：市场热度/增长补充 query

Agent：`analyst_agent`

使用 prompt：

- System：`brief_query_template_system_prompt`
- User instruction：`brief_query_template_instruction`

调用位置：

- `workflow.py` 的 `analyst_agent(...)`

输出：

- `DetailQueryPlan`
- 为候选竞品生成补充搜索 query。

### 6. Phase 2 QA：市场热度/增长信息质检

Agent：`qa_agent`

使用 prompt：

- User：`qa_agent_prompt_7`

调用位置：

- `workflow.py` 的 `analyst_agent(...)`

说明：

- `qa_agent_prompt_7` 内部已经拼入 `qa_system_prompt`，并通过 `.format(...)` 注入 `product_info` 和 `market_growth_info`。

输出：

- `MarketGrowthQAOutput`

### 7. Phase 2 分析：候选相关性评分

Agent：`analyst_agent`

使用 prompt：

- System：`competitor_relevance_score_system_prompt`

调用位置：

- `workflow.py` 的 `analyst_agent(...)`

输出：

- `CompetitorRelevanceScore`
- 后端归一化分数并生成最终候选选择结果。

### 8. Checkpoint 1：竞品阵型 QA

Agent：`qa_agent`

使用 prompt：

- System：`qa_system_prompt`
- User：`phase_2_qa_intent`

调用位置：

- `workflow.py` 的 `qa_agent(...)`

输出：

- `QAInspectionOutput`
- 通过则进入 Phase 3。
- 不通过则回到 Phase 2 分析。

### 9. Phase 3 规划：分析工具与采集策略

Agent：`plan_agent`

使用 prompt：

- System：`plan_system_prompt`
- User：`phase_3_plan_intent`

调用位置：

- `workflow.py` 的 `plan_agent(...)`

输出：

- `StateTreePatch`
- 更新 `state_tree.phase_3_strategy`
- 后端规则同时生成或补齐 `tool_plan` 和 `collection_plan`。

### 10. Phase 4 采集：资料卡 query 生成

Agent：`source_collector_agent`

使用 prompt：

- System：`detail_card_query_system_prompt`

调用位置：

- `card_collector.py`
- 由 `workflow.py` 调用 `collect_detail_cards(...)`

输出：

- `DetailQueryPlan`
- 竞品资料卡 query templates
- 行业资料卡 query

### 11. Phase 4 采集：chunk 级资料卡填充

Agent：`source_collector_agent`

使用 prompt：

- 竞品卡：`competitor_card_patch_system_prompt`
- 行业卡：`industry_card_patch_system_prompt`

调用位置：

- `card_collector.py`

输出：

- `FieldExtractionResult`
- 增量填充竞品资料卡和行业资料卡。

### 12. Phase 4 采集：资料卡 QA 与缺口补搜

Agent：`source_collector_agent`

使用 prompt：

- QA：`card_qa_system_prompt`
- 补搜：`card_gap_query_system_prompt`

调用位置：

- `card_collector.py`

输出：

- `CardQAResult`
- 缺口补搜 query
- 最终资料卡、清洗资料卡和报告引用。

### 13. Phase 4 工具执行

Agent：`tool_executor_agent`

使用 prompt：

- System：`tool_executor_system_prompt`
- User：`tool_executor_user_prompt`

调用位置：

- `tool_executor.py`
- 由 `workflow.py` 调用 `execute_tools(...)`

输出：

- `ToolExecutionResult`
- `tool_results`
- `tool_sandboxes`

### 14. Phase 4 分析综合

Agent：`analyst_agent`

当前说明：

- 此阶段不再使用 `structurer_agent.py`。
- 现有分析综合逻辑基于前序状态、资料卡、工具结果和规则兜底构建 `analysis_report`。
- `analyst_agent.py` 中当前实际被 LLM 调用的 prompt 仍主要集中在 Phase 2：
  - `brief_query_template_system_prompt`
  - `brief_query_template_instruction`
  - `competitor_relevance_score_system_prompt`

输出：

- `analysis_report`
- `state_tree.phase_4_synthesis`

### 15. 问卷设计

Agent：`survey_design_agent`

使用 prompt：

- System：`survey_design_system_prompt`
- User：`survey_design_user_prompt`

调用位置：

- `survey.py`
- 由 `workflow.py` 的 `survey_design_agent(...)` 调用

说明：

- 仅当 `workflow_phase == "survey_design"` 时进入。

输出：

- `SurveyDesign`

### 16. 问卷结果分析

使用 prompt：

- System：`survey_analysis_system_prompt`
- User：`survey_analysis_user_prompt`

调用位置：

- `survey.py` 的 `analyze_survey_responses(...)`

输出：

- `SurveyAnalysisResult`

说明：

- 这是问卷上传/分析链路使用的 prompt，不是默认竞品分析主干每次必跑节点。

### 17. Checkpoint 2：最终状态树 QA

Agent：`qa_agent`

使用 prompt：

- System：`qa_system_prompt`
- User：`phase_4_qa_intent`

调用位置：

- `workflow.py` 的 `qa_agent(...)`

输出：

- `QAInspectionOutput`
- 通过则进入 writer。
- 不通过则回到 Phase 4 synthesis。

### 18. Writer 分章节报告生成

Agent：`writer_agent`

使用 prompt：

- System：`writer_system_prompt`
- User section prompts：
  - `REPORT_TITLE_PROMPT`
  - `REPORT_BACKGROUND_PROMPT`
  - `REPORT_COMPETITOR_SELECTION_PROMPT`
  - `REPORT_TOOL_SECTION_PROMPT`
  - `REPORT_KEY_FINDINGS_PROMPT`
  - `REPORT_RECOMMENDATIONS_PROMPT`
  - `REPORT_FINAL_UNIFY_PROMPT`

调用位置：

- `workflow.py` 的 `writer_agent(...)`

输出：

- 最终 Markdown 报告
- `report.report_chapters`
- `state_tree.final_report_markdown`

## 当前导入但主流程未实际调用的 prompt

以下变量仍可保留为兼容或历史设计，但当前主流程没有实际传入 LLM 调用：

### `plan_agent.py`

- `phase_2_plan_intent`
- `phase_4_plan_intent`

### `analyst_agent.py`

- `analyst_system_prompt`
- `phase_2_analyst_intent`

说明：

- `analyst_agent.py` 文件不能删除，因为当前仍使用：
  - `brief_query_template_system_prompt`
  - `brief_query_template_instruction`
  - `competitor_relevance_score_system_prompt`

## 清理规则

如果继续清理 prompt，请优先按变量级别判断，不要只看文件名。

删除前至少搜索：

- 变量是否被传入 `_llm_text(...)`
- 变量是否被传入 `_llm_structured(...)`
- 变量是否被传入 `retry_structured_ainvoke(...)`
- 是否还有测试、前端展示名、模型枚举或兼容导出依赖

from __future__ import annotations

from myagent.prompt.analyst_agent import (
    analyst_system_prompt,
    brief_query_template_instruction,
    brief_query_template_system_prompt,
    competitor_relevance_score_system_prompt,
    phase_2_analyst_intent,
)
from myagent.prompt.card_agent import (
    card_gap_query_system_prompt,
    card_qa_system_prompt,
    competitor_card_patch_system_prompt,
    detail_card_query_system_prompt,
    industry_card_patch_system_prompt,
)
from myagent.prompt.plan_agent import (
    CURRENT_STATE_TREE_TEMPLATE,
    analysis_intent_system_prompt,
    analysis_intent_user_instruction,
    new_state_tree,
    phase_1_plan_intent,
    phase_2_plan_intent,
    phase_3_plan_intent,
    phase_4_plan_intent,
    plan_system_prompt,
)
from myagent.prompt.qa_agent import (
    candidate_existence_qa_instruction,
    candidate_existence_qa_system_prompt,
    qa_agent_prompt_7,
    qa_system_prompt,
    phase_2_qa_intent,
    phase_4_qa_intent,
)
from myagent.prompt.search_agent import (
    candidate_chunk_generation_prompt,
    candidate_query_generation_prompt,
)
from myagent.prompt.tool_executor_agent import tool_executor_system_prompt, tool_executor_user_prompt
from myagent.prompt.writer_agent import (
    REPORT_BACKGROUND_PROMPT,
    REPORT_COMPETITOR_SELECTION_PROMPT,
    REPORT_FINAL_UNIFY_PROMPT,
    REPORT_KEY_FINDINGS_PROMPT,
    REPORT_RECOMMENDATIONS_PROMPT,
    REPORT_TITLE_PROMPT,
    REPORT_TOOL_SECTION_PROMPT,
    writer_system_prompt,
)


__all__ = [
    "CURRENT_STATE_TREE_TEMPLATE",
    "new_state_tree",
    "plan_system_prompt",
    "analysis_intent_system_prompt",
    "analysis_intent_user_instruction",
    "phase_1_plan_intent",
    "phase_2_plan_intent",
    "phase_3_plan_intent",
    "phase_4_plan_intent",
    "candidate_query_generation_prompt",
    "candidate_chunk_generation_prompt",
    "detail_card_query_system_prompt",
    "competitor_card_patch_system_prompt",
    "industry_card_patch_system_prompt",
    "card_gap_query_system_prompt",
    "card_qa_system_prompt",
    "analyst_system_prompt",
    "brief_query_template_system_prompt",
    "brief_query_template_instruction",
    "competitor_relevance_score_system_prompt",
    "phase_2_analyst_intent",
    "qa_system_prompt",
    "qa_agent_prompt_7",
    "candidate_existence_qa_system_prompt",
    "candidate_existence_qa_instruction",
    "phase_2_qa_intent",
    "phase_4_qa_intent",
    "writer_system_prompt",
    "REPORT_TITLE_PROMPT",
    "REPORT_BACKGROUND_PROMPT",
    "REPORT_COMPETITOR_SELECTION_PROMPT",
    "REPORT_TOOL_SECTION_PROMPT",
    "REPORT_KEY_FINDINGS_PROMPT",
    "REPORT_RECOMMENDATIONS_PROMPT",
    "REPORT_FINAL_UNIFY_PROMPT",
    "tool_executor_system_prompt",
    "tool_executor_user_prompt",
]

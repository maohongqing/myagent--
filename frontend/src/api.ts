import type {
  AnalysisTask,
  CompetitorReport,
  CreateTaskRequest,
  DecisionAction,
  EvidenceItem,
  KnowledgeCardDetail,
  KnowledgeCompetitorCard,
  KnowledgeImportPreview,
  KnowledgeIndustryCard,
  KnowledgeUpdateSuggestion,
  SurveyAnalysisResult,
  SurveyDesign,
  SurveyResponseAnalysisRequest,
  SearchRunDetail,
  SearchRunList,
  TaskDetail,
  TaskOutputList
} from "./types";

export const API_BASE = import.meta.env.VITE_API_BASE ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || response.statusText);
  }
  return response.json();
}

export function createTask(payload: CreateTaskRequest) {
  return request<AnalysisTask>("/api/tasks", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function resumeLatestTask() {
  return request<AnalysisTask>("/api/tasks/resume-latest", {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function listTasks(limit = 50) {
  return request<AnalysisTask[]>(`/api/tasks?limit=${limit}`);
}

export function submitClarification(taskId: string, answer: string, selectedOption?: string) {
  return request<AnalysisTask>(`/api/tasks/${taskId}/clarifications`, {
    method: "POST",
    body: JSON.stringify({ answer, selected_option: selectedOption })
  });
}

export function submitDecision(taskId: string, action: DecisionAction) {
  return request<AnalysisTask>(`/api/tasks/${taskId}/decisions`, {
    method: "POST",
    body: JSON.stringify({ action })
  });
}

export function cancelTask(taskId: string) {
  return request<AnalysisTask>(`/api/tasks/${taskId}/cancel`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function getTask(taskId: string) {
  return request<TaskDetail>(`/api/tasks/${taskId}`);
}

export function getReport(taskId: string) {
  return request<CompetitorReport>(`/api/tasks/${taskId}/report`);
}

export function getSources(taskId: string) {
  return request<EvidenceItem[]>(`/api/tasks/${taskId}/sources`);
}

export function getSurveyDesign(taskId: string) {
  return request<SurveyDesign>(`/api/tasks/${taskId}/survey-design`);
}

export function regenerateSurveyDesign(taskId: string) {
  return request<SurveyDesign>(`/api/tasks/${taskId}/survey-design/regenerate`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function analyzeSurveyResponses(taskId: string, payload: SurveyResponseAnalysisRequest) {
  return request<SurveyAnalysisResult>(`/api/tasks/${taskId}/survey-responses/analyze`, {
    method: "POST",
    body: JSON.stringify(payload)
  });
}

export function getSurveyAnalysis(taskId: string) {
  return request<SurveyAnalysisResult>(`/api/tasks/${taskId}/survey-analysis`);
}

export function appendSurveyAnalysisToReport(taskId: string) {
  return request<CompetitorReport>(`/api/tasks/${taskId}/survey-analysis/append-to-report`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function getSearchRuns(taskId: string) {
  return request<SearchRunList>(`/api/tasks/${taskId}/search-runs`);
}

export function getSearchRun(taskId: string, runKey: string) {
  return request<SearchRunDetail>(`/api/tasks/${taskId}/search-runs/${runKey}`);
}

export function getTaskOutputs(taskId: string) {
  return request<TaskOutputList>(`/api/tasks/${taskId}/outputs`);
}

export function listKnowledgeCompetitors(query = "", industry = "") {
  const params = new URLSearchParams();
  if (query) params.set("q", query);
  if (industry) params.set("industry", industry);
  const suffix = params.toString();
  return request<KnowledgeCompetitorCard[]>(`/api/knowledge/competitors${suffix ? `?${suffix}` : ""}`);
}

export function listKnowledgeIndustries(query = "") {
  return request<KnowledgeIndustryCard[]>(`/api/knowledge/industries${query ? `?q=${encodeURIComponent(query)}` : ""}`);
}

export function getKnowledgeCompetitor(cardId: string) {
  return request<KnowledgeCardDetail<KnowledgeCompetitorCard>>(`/api/knowledge/competitors/${cardId}`);
}

export function getKnowledgeIndustry(cardId: string) {
  return request<KnowledgeCardDetail<KnowledgeIndustryCard>>(`/api/knowledge/industries/${cardId}`);
}

export function createKnowledgeCompetitor(card: KnowledgeCompetitorCard) {
  return request<KnowledgeCompetitorCard>("/api/knowledge/competitors", {
    method: "POST",
    body: JSON.stringify(card)
  });
}

export function saveKnowledgeCompetitor(card: KnowledgeCompetitorCard) {
  return request<KnowledgeCompetitorCard>(`/api/knowledge/competitors/${card.id}`, {
    method: "PUT",
    body: JSON.stringify(card)
  });
}

export function saveKnowledgeIndustry(card: KnowledgeIndustryCard) {
  return request<KnowledgeIndustryCard>(`/api/knowledge/industries/${card.id}`, {
    method: "PUT",
    body: JSON.stringify(card)
  });
}

export function deleteKnowledgeCompetitor(cardId: string) {
  return request<KnowledgeCompetitorCard>(`/api/knowledge/competitors/${cardId}`, {
    method: "DELETE"
  });
}

export function deleteKnowledgeIndustry(cardId: string) {
  return request<KnowledgeIndustryCard>(`/api/knowledge/industries/${cardId}`, {
    method: "DELETE"
  });
}

export function previewKnowledgeImport(taskId: string) {
  return request<KnowledgeImportPreview>(`/api/knowledge/import-from-task/${taskId}`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function commitKnowledgeImport(taskId: string, competitorCardIds: string[], industryCardIds: string[]) {
  return request<KnowledgeImportPreview>(`/api/knowledge/import-from-task/${taskId}/commit`, {
    method: "POST",
    body: JSON.stringify({ competitor_card_ids: competitorCardIds, industry_card_ids: industryCardIds })
  });
}

export function listKnowledgeSuggestions(status = "pending") {
  return request<KnowledgeUpdateSuggestion[]>(`/api/knowledge/suggestions?status=${encodeURIComponent(status)}`);
}

export function acceptKnowledgeSuggestion(suggestionId: string) {
  return request<KnowledgeUpdateSuggestion>(`/api/knowledge/suggestions/${suggestionId}/accept`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function rejectKnowledgeSuggestion(suggestionId: string) {
  return request<KnowledgeUpdateSuggestion>(`/api/knowledge/suggestions/${suggestionId}/reject`, {
    method: "POST",
    body: JSON.stringify({})
  });
}

export function exportUrl(taskId: string, format: "md" | "json" | "pdf" | "docx") {
  return `${API_BASE}/api/tasks/${taskId}/export?format=${format}`;
}

export function eventsUrl(taskId: string) {
  return `${API_BASE}/api/tasks/${taskId}/events`;
}

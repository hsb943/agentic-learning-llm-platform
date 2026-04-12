import {
  LearningCheckpoint,
  QuestionResponse,
  SessionStateResponse,
} from "./types";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(detail || `Request failed with status ${response.status}`);
  }

  return response.json() as Promise<T>;
}

export function startSession(payload: {
  topic: string;
  goals: string[];
  note: string;
}) {
  return request<SessionStateResponse>("/sessions/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export function confirmCheckpoints(
  threadId: string,
  checkpoints: LearningCheckpoint[],
) {
  return request<SessionStateResponse>(
    `/sessions/${threadId}/checkpoints/confirm`,
    {
      method: "POST",
      body: JSON.stringify({ checkpoints }),
    },
  );
}

export function getCurrentQuestion(threadId: string) {
  return request<QuestionResponse>(`/sessions/${threadId}/question`);
}

export function submitAnswer(threadId: string, answer: string) {
  return request<SessionStateResponse>(`/sessions/${threadId}/answer`, {
    method: "POST",
    body: JSON.stringify({ answer }),
  });
}

export function getSession(threadId: string) {
  return request<SessionStateResponse>(`/sessions/${threadId}`);
}

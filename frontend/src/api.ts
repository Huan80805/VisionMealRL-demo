import type {
  CreateSessionPayload,
  CreateSessionResponse,
  PreferenceTemplate,
  SelectMealResponse,
} from "./types";

const API_BASE = import.meta.env.VITE_AGENT_API_BASE || "/agent-api";

export function apiAssetUrl(path: string): string {
  if (/^https?:\/\//.test(path)) {
    return path;
  }
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

async function requestJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });

  const text = await response.text();
  const payload = text ? JSON.parse(text) : null;
  if (!response.ok) {
    const message =
      payload && typeof payload.error === "string"
        ? payload.error
        : `Request failed with ${response.status}`;
    throw new Error(message);
  }
  return payload as T;
}

export async function getHealth(): Promise<{ ok: boolean }> {
  return requestJson<{ ok: boolean }>("/api/health");
}

export async function getPreferenceTemplates(limit = 128): Promise<PreferenceTemplate[]> {
  const payload = await requestJson<{ templates: PreferenceTemplate[] }>(
    `/api/preference-templates?limit=${limit}`,
  );
  return payload.templates;
}

export async function getHorizons(): Promise<number[]> {
  const payload = await requestJson<{ horizons: number[] }>("/api/horizons");
  return payload.horizons;
}

export async function createSession(
  payload: CreateSessionPayload,
): Promise<CreateSessionResponse> {
  return requestJson<CreateSessionResponse>("/api/session", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

export async function selectMeal(
  sessionId: string,
  action: number,
  topK = 4,
): Promise<SelectMealResponse> {
  return requestJson<SelectMealResponse>(`/api/session/${sessionId}/select`, {
    method: "POST",
    body: JSON.stringify({ action, top_k: topK }),
  });
}

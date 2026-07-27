export const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE ??
  "http://127.0.0.1:8000/api"
).replace(/\/$/, "");

export async function requestJson<T>(
  path: string,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init);

  if (!response.ok) {
    throw new Error(await response.text());
  }

  return response.json() as Promise<T>;
}

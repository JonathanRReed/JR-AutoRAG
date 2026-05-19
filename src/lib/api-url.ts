export const API_PROXY_PREFIX = "/__api";

export function normalizeApiBaseUrl(value?: string) {
  const trimmed = value?.trim().replace(/\/+$/, "");
  return trimmed || API_PROXY_PREFIX;
}

export function buildApiUrl(baseUrl: string | undefined, path: string) {
  const root = normalizeApiBaseUrl(baseUrl);
  const route = path.startsWith("/") ? path : `/${path}`;
  return `${root}${route}`;
}

export function resolveDefaultApiBaseUrl() {
  const envBase =
    (import.meta.env?.BUN_PUBLIC_BROWSER_API_BASE_URL as string | undefined) ||
    (import.meta.env?.VITE_BROWSER_API_BASE_URL as string | undefined);
  if (envBase) {
    return normalizeApiBaseUrl(envBase.replace("http://localhost:8000", "http://127.0.0.1:8000"));
  }
  return API_PROXY_PREFIX;
}

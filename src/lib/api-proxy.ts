const DEFAULT_API_BASE_URL = "http://127.0.0.1:8000";

const SENSITIVE_API_PREFIXES = [
  "/api/traces",
  "/api/artifacts",
  "/api/cache",
];

const PUBLIC_MANAGEMENT_STATUS_PATHS = new Set([
  "/api/artifacts/status",
  "/api/cache/status",
]);

export const getApiBaseUrl = (): string => (
  process.env.BUN_PUBLIC_API_BASE_URL ||
  process.env.VITE_API_BASE_URL ||
  DEFAULT_API_BASE_URL
).replace(/\/+$/, "");

export const isApiAuthEnabled = (): boolean => (
  ["true", "1", "yes"].includes((process.env.AUTORAG_AUTH_ENABLED || "false").toLowerCase())
);

export const getBackendPath = (requestPath: string, prefix: string): string => (
  prefix && requestPath.startsWith(prefix)
    ? requestPath.slice(prefix.length) || "/"
    : requestPath
);

export const isSensitiveManagementApiPath = (path: string): boolean => {
  const normalizedPath = path.replace(/\/+$/, "") || "/";

  if (PUBLIC_MANAGEMENT_STATUS_PATHS.has(normalizedPath)) {
    return false;
  }

  return SENSITIVE_API_PREFIXES.some((prefix) => (
    normalizedPath === prefix || normalizedPath.startsWith(`${prefix}/`)
  ));
};

export const isProxyRequestAllowed = (backendPath: string): boolean => (
  isApiAuthEnabled() || !isSensitiveManagementApiPath(backendPath)
);

export const proxyApiRequest = async (req: Request, prefix: string): Promise<Response> => {
  const url = new URL(req.url);
  const path = getBackendPath(url.pathname, prefix);

  if (!isProxyRequestAllowed(path)) {
    return Response.json(
      {
        detail: (
          "This management API is not available through the unauthenticated UI proxy. " +
          "Enable AUTORAG_AUTH_ENABLED=true and configure AUTORAG_API_KEYS before exposing it."
        ),
      },
      { status: 403 },
    );
  }

  try {
    return await fetch(getApiBaseUrl() + path + url.search, {
      method: req.method,
      headers: req.headers,
      body: req.body,
    });
  } catch {
    return new Response("Backend execution failed", { status: 502 });
  }
};

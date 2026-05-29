import { describe, expect, test } from "bun:test";

import { DEFAULT_API_BASE_URL, buildApiUrl, normalizeApiBaseUrl } from "./api-url";

describe("api url helpers", () => {
  test("defaults to the local FastAPI origin instead of a same-origin backend proxy", () => {
    expect(normalizeApiBaseUrl("")).toBe(DEFAULT_API_BASE_URL);
    expect(normalizeApiBaseUrl(undefined)).toBe(DEFAULT_API_BASE_URL);
  });

  test("joins direct API origins and paths", () => {
    expect(buildApiUrl("http://127.0.0.1:8018/", "/healthz")).toBe("http://127.0.0.1:8018/healthz");
  });

  test("joins configured API bases and backend routes without rewriting paths", () => {
    expect(buildApiUrl("http://127.0.0.1:8000", "/config")).toBe("http://127.0.0.1:8000/config");
    expect(buildApiUrl("http://127.0.0.1:8000", "/api/artifacts/status")).toBe("http://127.0.0.1:8000/api/artifacts/status");
  });
});

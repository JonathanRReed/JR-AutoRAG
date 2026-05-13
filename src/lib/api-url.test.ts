import { describe, expect, test } from "bun:test";

import { API_PROXY_PREFIX, buildApiUrl, normalizeApiBaseUrl } from "./api-url";

describe("api url helpers", () => {
  test("defaults to the same-origin backend proxy", () => {
    expect(normalizeApiBaseUrl("")).toBe(API_PROXY_PREFIX);
    expect(normalizeApiBaseUrl(undefined)).toBe(API_PROXY_PREFIX);
  });

  test("joins direct API origins and paths", () => {
    expect(buildApiUrl("http://127.0.0.1:8018/", "/healthz")).toBe("http://127.0.0.1:8018/healthz");
  });

  test("joins proxy base and backend routes without hiding API-prefixed paths", () => {
    expect(buildApiUrl(API_PROXY_PREFIX, "/config")).toBe("/__api/config");
    expect(buildApiUrl(API_PROXY_PREFIX, "/api/artifacts/status")).toBe("/__api/api/artifacts/status");
  });
});

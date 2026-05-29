import { afterEach, describe, expect, test } from "bun:test";

import {
  getBackendPath,
  isProxyRequestAllowed,
  isSensitiveManagementApiPath,
  proxyApiRequest,
} from "./api-proxy";

afterEach(() => {
  delete process.env.AUTORAG_AUTH_ENABLED;
});

describe("API proxy management-route guard", () => {
  test("normalizes /__api requests before applying management API checks", () => {
    expect(getBackendPath("/__api/api/cache/clear", "/__api")).toBe("/api/cache/clear");
    expect(getBackendPath("/api/cache/clear", "")).toBe("/api/cache/clear");
  });

  test("identifies sensitive trace, artifact, and cache management APIs", () => {
    expect(isSensitiveManagementApiPath("/api/traces/download")).toBe(true);
    expect(isSensitiveManagementApiPath("/api/artifacts/graph")).toBe(true);
    expect(isSensitiveManagementApiPath("/api/artifacts/build")).toBe(true);
    expect(isSensitiveManagementApiPath("/api/cache/clear")).toBe(true);
    expect(isSensitiveManagementApiPath("/api/cache/rebuild")).toBe(true);
  });

  test("allows status APIs and unrelated application APIs", () => {
    expect(isSensitiveManagementApiPath("/api/artifacts/status")).toBe(false);
    expect(isSensitiveManagementApiPath("/api/cache/status")).toBe(false);
    expect(isSensitiveManagementApiPath("/query")).toBe(false);
  });

  test("blocks sensitive APIs while backend API-key auth is disabled", () => {
    process.env.AUTORAG_AUTH_ENABLED = "false";

    expect(isProxyRequestAllowed("/api/traces/last")).toBe(false);
    expect(isProxyRequestAllowed("/api/cache/clear")).toBe(false);
    expect(isProxyRequestAllowed("/api/artifacts/status")).toBe(true);
  });

  test("allows sensitive APIs to reach FastAPI when backend API-key auth is enabled", () => {
    process.env.AUTORAG_AUTH_ENABLED = "true";

    expect(isProxyRequestAllowed("/api/traces/last")).toBe(true);
    expect(isProxyRequestAllowed("/api/cache/clear")).toBe(true);

    process.env.AUTORAG_AUTH_ENABLED = "1";
    expect(isProxyRequestAllowed("/api/artifacts/graph")).toBe(true);
  });

  test("returns 403 before proxying sensitive APIs without backend API-key auth", async () => {
    process.env.AUTORAG_AUTH_ENABLED = "false";

    const legacyProxyResponse = await proxyApiRequest(
      new Request("http://ui.example/api/cache/clear?include_disk=true", { method: "DELETE" }),
      "",
    );
    const prefixedProxyResponse = await proxyApiRequest(
      new Request("http://ui.example/__api/api/traces/download"),
      "/__api",
    );

    expect(legacyProxyResponse.status).toBe(403);
    expect(prefixedProxyResponse.status).toBe(403);
  });
});

import { expect, request, test } from "@playwright/test";

const apiBaseURL = process.env.JR_E2E_API_BASE_URL ?? "http://127.0.0.1:8129";
const apiKey = process.env.JR_E2E_API_KEY ?? "interview-smoke-key";

test.describe("interview demo smoke", () => {
  test.beforeEach(async ({ page }) => {
    const api = await request.newContext({
      baseURL: apiBaseURL,
      extraHTTPHeaders: { "X-API-Key": apiKey },
    });
    await api.delete("/onboarding/demo");
    const seed = await api.post("/onboarding/demo/seed");
    expect(seed.ok()).toBeTruthy();
    await api.dispose();

    await page.addInitScript((key) => {
      window.sessionStorage.setItem("jr-autorag-api-key", key);
    }, apiKey);
  });

  test("loads demo corpus, queries with evidence, and shows readiness surfaces", async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on("console", (message) => {
      if (message.type() === "error") {
        consoleErrors.push(message.text());
      }
    });

    await page.goto("/");

    await expect(page.getByRole("heading", { name: "JR AutoRAG" })).toBeVisible();
    await expect(page.getByText("FastAPI reachable")).toBeVisible();
    await expect(page.getByText(/documents indexed/i)).toBeVisible();
    await expect(page.getByText(/Security posture is ready/i)).toBeVisible();
    await expect(page.getByText("Security Posture", { exact: true })).toBeVisible();
    await expect(page.getByText(/client ready|local only/i).first()).toBeVisible();

    await page.getByRole("tab", { name: /Query/i }).click();
    await page.getByLabel("Ask a grounded question").fill(
      "What local-only controls, authentication settings, and handoff receipts are required before a client install?"
    );
    await page.getByRole("button", { name: "Ask grounded question" }).click();

    await expect(page.getByText(/Query complete/i)).toBeVisible({ timeout: 25_000 });
    await expect(page.getByRole("heading", { name: "Sources", exact: true })).toBeVisible();
    await expect(page.getByRole("heading", { name: /Sources \(/ })).toBeVisible();
    await expect(page.getByRole("button", { name: "Citation 1 - jump to source" })).toBeVisible();
    await expect(page.getByText(/local auth receipt/i).first()).toBeVisible();

    await page.getByRole("tab", { name: /Quality/i }).click();
    await expect(page.getByRole("heading", { name: "Quality Cockpit" })).toBeVisible();
    await expect(page.getByText("Evaluation Evidence")).toBeVisible();

    const errors = await page.evaluate(() =>
      performance.getEntriesByType("resource")
        .filter((entry) => entry.name.includes("/undefined"))
        .map((entry) => entry.name)
    );
    expect(errors).toEqual([]);
    expect(consoleErrors).toEqual([]);
  });
});

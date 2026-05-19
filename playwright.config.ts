import { defineConfig, devices } from "@playwright/test";

const webBaseURL = process.env.JR_E2E_WEB_BASE_URL ?? "http://127.0.0.1:3210";

export default defineConfig({
  testDir: "./e2e",
  testMatch: /.*\.pw\.ts/,
  timeout: 60_000,
  expect: {
    timeout: 12_000,
  },
  fullyParallel: false,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [["list"], ["html", { open: "never", outputFolder: "playwright-report" }]] : "list",
  use: {
    baseURL: webBaseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
    video: "retain-on-failure",
  },
  outputDir: "output/playwright",
  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],
});

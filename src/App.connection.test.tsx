import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, afterEach, beforeEach, describe, expect, test } from "bun:test";

const UI_ORIGIN = "http://localhost:3000";
GlobalRegistrator.register({ url: `${UI_ORIGIN}/` });
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const { act, createElement } = await import("react");
const { createRoot } = await import("react-dom/client");
const { App } = await import("./App");
const { ToastProvider } = await import("@/components/ui/toast");

// Synthetic values only. No real key or provider is ever contacted: every
// request goes to the fake fetch below and unknown origins fail like a dead host.
const ORIGIN_A = "http://127.0.0.1:8000";
const ORIGIN_B = "http://api-b.test:9000";
const GOOD_KEY = "synthetic-good-key";
const WRONG_KEY = "synthetic-wrong-key";
const READ_KEY = "synthetic-read-only-key";

type Logged = { origin: string; path: string; key: string | null; signal?: AbortSignal };
// A gate holds every request to its path until the test settles it. Like a real
// fetch, the fake refuses a request whose signal is already aborted. It ignores
// an abort that happens mid-flight on purpose, so a late response still arrives
// after the app has moved on and the app's own guards are what get tested.
// `key` has every scope; `readKey` only "read", like api/app/core/security_middleware.py.
type Server = {
  health: number;
  key: string | null;
  readKey: string | null;
  provider: string;
  docs: unknown[];
  gates: Record<string, Promise<unknown>>;
};
const ADMIN_PATHS = ["/config", "/monitoring/traces", "/providers/local"];

let serverA: Server;
let serverB: Server;
let log: Logged[];
const realFetch = globalThis.fetch;
const realSetInterval = globalThis.setInterval;
const HEARTBEAT_MS = 20000;
let heartbeat: (() => void) | undefined;

const json = (status: number, body: unknown) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const configFor = (provider: string) => ({
  profile: "Default",
  deployment_profile: "local",
  provider: { name: provider },
  provider_profiles: [],
  retrieval: {},
  ingest: {},
  backends: {},
  fallbacks: {},
});

const protectedBody = (path: string, server: Server) => {
  if (path === "/config") return configFor(server.provider);
  if (path === "/monitoring/cache") return {};
  if (path === "/security/posture") return { level: "local_only", summary: "synthetic", settings: {}, checks: [] };
  if (path === "/onboarding") {
    return {
      flow: { steps: [], current_step: 0, progress: 0, is_complete: false },
      demo_mode: false,
      demo_seeded: false,
      document_count: 0,
      demo_document_count: 0,
      sample_documents: [],
      example_queries: [],
    };
  }
  if (path === "/documents") return server.docs;
  if (["/monitoring/traces", "/providers/local"].includes(path)) return [];
  return undefined;
};

const fakeFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  if (init?.signal?.aborted) throw new DOMException("The operation was aborted.", "AbortError");
  // A URL without a scheme resolves against the page, like a browser fetch.
  const url = new URL(String(input), location.href);
  const key = new Headers(init?.headers).get("X-API-Key");
  log.push({ origin: url.origin, path: url.pathname, key, signal: init?.signal ?? undefined });
  // The UI server answers every unmatched path with index.html (src/index.ts "/*").
  if (url.origin === UI_ORIGIN) return new Response("<!doctype html>", { status: 200, headers: { "Content-Type": "text/html" } });
  const server = url.origin === ORIGIN_A ? serverA : url.origin === ORIGIN_B ? serverB : undefined;
  if (!server) throw new TypeError("Failed to fetch");
  await server.gates[url.pathname];
  if (url.pathname === "/healthz") return json(server.health, { status: server.health < 400 ? "ok" : "error" });
  if (server.key !== null && key === null) return json(401, { detail: "API key required. Provide X-API-Key header." });
  const readOnly = key !== null && key === server.readKey && !ADMIN_PATHS.includes(url.pathname);
  if (server.key !== null && key !== server.key && !readOnly) return json(403, { detail: "Invalid API key or insufficient permissions." });
  const body = protectedBody(url.pathname, server);
  return body === undefined ? json(404, { detail: "Not modeled by this test" }) : json(200, body);
}) as typeof fetch;

let container: HTMLElement;
let root: ReturnType<typeof createRoot> | null;

const settle = async () => {
  for (let i = 0; i < 8; i += 1) {
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, 0));
    });
  }
};

const mount = async () => {
  container = document.createElement("div");
  document.body.appendChild(container);
  root = createRoot(container);
  await act(async () => {
    root!.render(createElement(ToastProvider, null, createElement(App)));
  });
  await settle();
};

const input = (label: string) => container.querySelector(`input[aria-label="${label}"]`) as HTMLInputElement;
const connectButton = () => input("API base URL").form!.querySelector('button[type="submit"]') as HTMLButtonElement;
const shown = (label: string) => {
  const dt = [...container.querySelectorAll("dt")].find(node => node.textContent === label);
  return dt?.nextElementSibling?.textContent ?? null;
};
const providerShown = () => shown("Provider");

const type = async (label: string, value: string) => {
  const element = input(label);
  const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value")!.set!;
  await act(async () => {
    setter.call(element, value);
    element.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await settle();
};

const clickConnect = async () => {
  await act(async () => {
    input("API base URL").form!.dispatchEvent(new Event("submit", { bubbles: true, cancelable: true }));
  });
  await settle();
};

const since = (mark: number) => log.slice(mark);

const toasts = () =>
  [...container.querySelectorAll('button[aria-label="Dismiss notification"]')].map(button =>
    [...button.parentElement!.querySelectorAll("p")].map(p => p.textContent).join(": "),
  );

const deferred = () => {
  let resolve!: () => void;
  const promise = new Promise<void>(res => {
    resolve = res;
  });
  return { promise, resolve };
};

const release = async (gate: { resolve: () => void }) => {
  gate.resolve();
  await settle();
};

// Fire one tick without awaiting it, so a tick held at a gate can stay pending.
const tickHeartbeat = async () => {
  await act(async () => {
    heartbeat!();
  });
  await settle();
};

const storeSessionKey = (key: string, origin = ORIGIN_A) => {
  sessionStorage.setItem("jr-autorag-api-key", key);
  sessionStorage.setItem("jr-autorag-api-key-origin", origin);
};

beforeEach(() => {
  sessionStorage.clear();
  localStorage.clear();
  log = [];
  serverA = { health: 200, key: GOOD_KEY, readKey: null, provider: "Provider A", docs: [], gates: {} };
  serverB = { health: 200, key: GOOD_KEY, readKey: null, provider: "Provider B", docs: [], gates: {} };
  globalThis.fetch = fakeFetch;
  heartbeat = undefined;
  globalThis.setInterval = ((handler: () => void, ms?: number) => {
    if (ms === HEARTBEAT_MS) heartbeat = handler;
    return realSetInterval(handler, ms);
  }) as typeof setInterval;
});

afterEach(async () => {
  if (root) {
    await act(async () => root!.unmount());
    root = null;
  }
  container?.remove();
  globalThis.fetch = realFetch;
  globalThis.setInterval = realSetInterval;
});

afterAll(async () => {
  await GlobalRegistrator.unregister();
});

describe("API connection lifecycle", () => {
  test("editing the URL does not contact the new origin until Connect", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    const mark = log.length;

    await type("API base URL", "http://api-b.te");
    await type("API base URL", ORIGIN_B);

    expect(since(mark).map(entry => entry.origin)).not.toContain(ORIGIN_B);
    expect(since(mark).map(entry => entry.origin)).not.toContain("http://api-b.te");
  });

  test("the stored key is not carried to a different origin", async () => {
    storeSessionKey(GOOD_KEY);
    serverB.key = null;
    await mount();

    await type("API base URL", ORIGIN_B);
    expect(input("API key").value).toBe("");
    const mark = log.length;
    await clickConnect();

    const toB = since(mark).filter(entry => entry.origin === ORIGIN_B);
    expect(toB.length).toBeGreaterThan(0);
    expect(toB.every(entry => entry.key === null)).toBe(true);
    expect(sessionStorage.getItem("jr-autorag-api-key")).toBeNull();
    expect(providerShown()).toBe("Provider B");
  });

  test("returning the draft URL to the applied origin restores the applied key", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();

    await type("API base URL", ORIGIN_B);
    expect(input("API key").value).toBe("");
    await type("API key", "synthetic-key-typed-for-b");
    await type("API base URL", `${ORIGIN_A}/`);
    expect(input("API key").value).toBe(GOOD_KEY);
  });

  test("a failed Connect keeps the previous connection applied", async () => {
    storeSessionKey(GOOD_KEY);
    serverB.health = 500;
    await mount();

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    const mark = log.length;
    await clickConnect();

    expect(since(mark).filter(entry => entry.origin === ORIGIN_B).map(entry => entry.path)).toEqual(["/healthz"]);
    expect(since(mark).every(entry => entry.key === null)).toBe(true);
    expect(providerShown()).toBe("Provider A");
    expect(sessionStorage.getItem("jr-autorag-api-key-origin")).toBe(ORIGIN_A);
  });

  test("a session key saved for another origin is not reused on reload", async () => {
    storeSessionKey(GOOD_KEY, ORIGIN_B);
    await mount();

    expect(log.length).toBeGreaterThan(0);
    expect(log.every(entry => entry.key === null)).toBe(true);
    expect(input("API key").value).toBe("");
  });

  test("editing the key does not send it until Connect", async () => {
    storeSessionKey(WRONG_KEY);
    await mount();
    const mark = log.length;

    await type("API key", GOOD_KEY);

    expect(since(mark).some(entry => entry.key === GOOD_KEY)).toBe(false);
    expect(sessionStorage.getItem("jr-autorag-api-key")).toBe(WRONG_KEY);
  });

  test("protected data loads after the key is corrected and Connect is clicked", async () => {
    storeSessionKey(WRONG_KEY);
    await mount();
    expect(providerShown()).toBe("No provider selected");

    await type("API key", GOOD_KEY);
    const mark = log.length;
    await clickConnect();

    const protectedWithGoodKey = since(mark).filter(entry => entry.path === "/config" && entry.key === GOOD_KEY);
    expect(protectedWithGoodKey.length).toBeGreaterThan(0);
    expect(providerShown()).toBe("Provider A");
    expect(sessionStorage.getItem("jr-autorag-api-key")).toBe(GOOD_KEY);
    expect(sessionStorage.getItem("jr-autorag-api-key-origin")).toBe(ORIGIN_A);
  });

  test("HTTP 500 from /healthz is not reported as connected", async () => {
    serverA.health = 500;
    await mount();

    expect(connectButton().textContent).toBe("Connect");
  });

  test("a late response from the previous endpoint does not overwrite the new one", async () => {
    storeSessionKey(GOOD_KEY);
    let releaseA!: () => void;
    serverA.gates["/config"] = new Promise<void>(resolve => {
      releaseA = resolve;
    });
    await mount();
    expect(providerShown()).toBe("No provider selected");

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    expect(providerShown()).toBe("Provider B");

    releaseA();
    await settle();
    expect(providerShown()).toBe("Provider B");
  });

  test("unmount aborts in-flight protected requests and heartbeat checks", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.gates["/config"] = new Promise<void>(() => {});
    await mount();
    const pending = log.filter(entry => entry.path === "/config");
    expect(pending.length).toBeGreaterThan(0);

    await act(async () => root!.unmount());
    root = null;

    expect(pending.every(entry => entry.signal?.aborted === true)).toBe(true);
    const heartbeats = log.filter(entry => entry.path === "/healthz");
    expect(heartbeats.length).toBeGreaterThan(0);
    expect(heartbeats.every(entry => entry.signal?.aborted === true)).toBe(true);
  });
});

describe("Connect attempts", () => {
  const startConnectToB = async (key: string) => {
    await type("API base URL", ORIGIN_B);
    await type("API key", key);
    const probe = deferred();
    serverB.gates["/healthz"] = probe.promise;
    await clickConnect();
    return probe;
  };

  test("a superseded Connect never applies, even when its probe answers last", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();

    const first = await startConnectToB(GOOD_KEY);
    const second = deferred();
    serverB.gates["/healthz"] = second.promise;
    await clickConnect();
    serverB.health = 500;
    await release(second);
    serverB.health = 200;
    await release(first);

    expect(providerShown()).toBe("Provider A");
    expect(connectButton().textContent).toBe("Connected");
    expect(sessionStorage.getItem("jr-autorag-api-key-origin")).toBe(ORIGIN_A);
    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_B} returned HTTP 500 for /healthz. Still using ${ORIGIN_A}.`]);
  });

  test("unmount during a pending Connect aborts the probe and sends nothing more", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();

    const probe = await startConnectToB(GOOD_KEY);
    await act(async () => root!.unmount());
    root = null;
    await release(probe);

    const toB = log.filter(entry => entry.origin === ORIGIN_B);
    expect(toB.map(entry => [entry.path, entry.signal?.aborted])).toEqual([["/healthz", true]]);
  });

  test("editing the draft while Connect is pending cancels that attempt and keeps the newer draft", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();

    const probe = await startConnectToB(GOOD_KEY);
    expect(connectButton().textContent).toBe("Connecting");
    await type("API key", "synthetic-newer-draft");
    expect(connectButton().textContent).toBe("Connected");
    await release(probe);

    expect(input("API base URL").value).toBe(ORIGIN_B);
    expect(input("API key").value).toBe("synthetic-newer-draft");
    expect(providerShown()).toBe("Provider A");
    expect(sessionStorage.getItem("jr-autorag-api-key-origin")).toBe(ORIGIN_A);
    expect(toasts()).toEqual([]);
  });

  test("a late success from a replaced connection does not mark the current one connected", async () => {
    storeSessionKey(GOOD_KEY);
    const staleConfig = deferred();
    serverA.gates["/config"] = staleConfig.promise;
    await mount();

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    expect(connectButton().textContent).toBe("Connected");
    serverB.health = 503;
    await tickHeartbeat();
    expect(connectButton().textContent).toBe("Connect");
    await release(staleConfig);

    expect(connectButton().textContent).toBe("Connect");
    expect(providerShown()).toBe("Provider B");
  });

  test("a wrong key is not applied, the previous connection stays, and the failure names the attempt", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();

    await type("API base URL", ORIGIN_B);
    await type("API key", WRONG_KEY);
    await clickConnect();

    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider A");
    expect(sessionStorage.getItem("jr-autorag-api-key")).toBe(GOOD_KEY);
    expect(sessionStorage.getItem("jr-autorag-api-key-origin")).toBe(ORIGIN_A);
    expect(log.filter(entry => entry.origin === ORIGIN_B).map(entry => [entry.path, entry.key])).toEqual([
      ["/healthz", null],
      ["/security/posture", WRONG_KEY],
    ]);
    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_B} rejected the API key (HTTP 403). Still using ${ORIGIN_A}.`]);
  });

  test("a failed Connect says whether the API was unreachable, unhealthy, or rejected the key", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    const attempts = [
      { url: "http://dead.test:1", key: "", health: 200, reason: "is unreachable" },
      { url: ORIGIN_B, key: GOOD_KEY, health: 500, reason: "returned HTTP 500 for /healthz" },
      { url: ORIGIN_B, key: "", health: 200, reason: "rejected the API key (HTTP 401)" },
    ];

    for (const attempt of attempts) {
      serverB.health = attempt.health;
      await type("API base URL", attempt.url);
      await type("API key", attempt.key);
      await clickConnect();
      expect(toasts().at(-1)).toBe(`Connect failed: ${attempt.url} ${attempt.reason}. Still using ${ORIGIN_A}.`);
    }
    expect(connectButton().textContent).toBe("Connected");
  });

  test("a stored key the server rejects is not shown as connected", async () => {
    storeSessionKey(WRONG_KEY);
    await mount();

    expect(log.some(entry => entry.path === "/security/posture" && entry.key === WRONG_KEY)).toBe(true);
    expect(connectButton().textContent).toBe("Connect");
  });

  test("Connected follows the latest heartbeat, not data loaded earlier", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    expect(connectButton().textContent).toBe("Connected");

    serverA.health = 503;
    await tickHeartbeat();

    expect(connectButton().textContent).toBe("Connect");
    expect(providerShown()).toBe("Provider A");
  });

  test("an unhealthy API is not shown as connected even when its data loads", async () => {
    serverA.health = 500;
    serverA.key = null;
    await mount();

    expect(providerShown()).toBe("Provider A");
    expect(connectButton().textContent).toBe("Connect");
  });

  test("a failed retry of the current connection clears Connected", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    expect(connectButton().textContent).toBe("Connected");

    serverA.health = 503;
    await clickConnect();

    expect(connectButton().textContent).toBe("Connect");
    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_A} returned HTTP 503 for /healthz.`]);
  });
});

describe("Status ordering", () => {
  // Each heartbeat tick and each Connect is a probe. Tests hold probes at
  // /healthz and release them out of order; the response is built on release.
  const holdHealth = (server: Server) => {
    const gate = deferred();
    server.gates["/healthz"] = gate.promise;
    return gate;
  };

  test.each([
    { older: 200, newer: 503, label: "Connect" },
    { older: 503, newer: 200, label: "Connected" },
  ])("an older heartbeat answering last ($older) does not overwrite a newer one ($newer)", async ({ older, newer, label }) => {
    storeSessionKey(GOOD_KEY);
    await mount();
    const first = holdHealth(serverA);
    await tickHeartbeat();
    const second = holdHealth(serverA);
    await tickHeartbeat();

    serverA.health = newer;
    await release(second);
    expect(connectButton().textContent).toBe(label);
    serverA.health = older;
    await release(first);

    expect(connectButton().textContent).toBe(label);
  });

  test("an older heartbeat success does not undo a failed retry of the same connection", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    const tick = holdHealth(serverA);
    await tickHeartbeat();

    delete serverA.gates["/healthz"];
    serverA.health = 503;
    await clickConnect();
    expect(connectButton().textContent).toBe("Connect");
    serverA.health = 200;
    await release(tick);

    expect(connectButton().textContent).toBe("Connect");
    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_A} returned HTTP 503 for /healthz.`]);
  });

  test("an older heartbeat failure does not undo a successful retry of the same connection", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.health = 503;
    await mount();
    expect(connectButton().textContent).toBe("Connect");
    const tick = holdHealth(serverA);
    await tickHeartbeat();

    delete serverA.gates["/healthz"];
    serverA.health = 200;
    await clickConnect();
    expect(connectButton().textContent).toBe("Connected");
    serverA.health = 503;
    await release(tick);

    expect(connectButton().textContent).toBe("Connected");
  });

  test("a retry that answers after a newer heartbeat does not overwrite it", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    const retry = holdHealth(serverA);
    await clickConnect();
    delete serverA.gates["/healthz"];
    await tickHeartbeat();
    expect(connectButton().textContent).toBe("Connecting");

    serverA.health = 503;
    await release(retry);

    expect(connectButton().textContent).toBe("Connected");
    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_A} returned HTTP 503 for /healthz.`]);
  });

  test("changing to a read-only key at the same URL hides data loaded with the admin key", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.readKey = READ_KEY;
    serverA.docs = [{ id: "doc-1", title: "Synthetic admin-era doc", text: "", metadata: {} }];
    await mount();
    expect(providerShown()).toBe("Provider A");
    expect(shown("Documents")).toBe("1");

    await type("API key", READ_KEY);
    await clickConnect();

    expect(connectButton().textContent).toBe("Connected");
    expect(sessionStorage.getItem("jr-autorag-api-key")).toBe(READ_KEY);
    expect(log.some(entry => entry.path === "/config" && entry.key === READ_KEY)).toBe(true);
    expect(providerShown()).toBe("No provider selected");
    expect(shown("Documents")).toBe("0");
  });

  test("a rejected key at the same URL keeps the applied connection and its data", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.docs = [{ id: "doc-1", title: "Synthetic doc", text: "", metadata: {} }];
    await mount();

    await type("API key", WRONG_KEY);
    await clickConnect();

    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider A");
    expect(shown("Documents")).toBe("1");
    expect(sessionStorage.getItem("jr-autorag-api-key")).toBe(GOOD_KEY);
    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_A} rejected the API key (HTTP 403). Still using ${ORIGIN_A}.`]);
  });

  test("a URL without http:// or https:// is refused, not reported as connected", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    const mark = log.length;

    await type("API base URL", "127.0.0.1:8000");
    await clickConnect();

    expect(since(mark).filter(entry => entry.origin === UI_ORIGIN).map(entry => entry.path)).toEqual([]);
    expect(toasts()).toEqual([`Connect failed: 127.0.0.1:8000 is not an http:// or https:// URL. Still using ${ORIGIN_A}.`]);
    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider A");
  });
});

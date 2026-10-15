import { GlobalRegistrator } from "@happy-dom/global-registrator";
import { afterAll, afterEach, beforeEach, describe, expect, test } from "bun:test";

const UI_ORIGIN = "http://localhost:3000";
GlobalRegistrator.register({ url: `${UI_ORIGIN}/` });
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

const { act, createElement } = await import("react");
const { createRoot } = await import("react-dom/client");
const appModule = await import("./App");
const { App } = appModule;
const setConnectionCheckTimeoutMsForTests =
  appModule.setConnectionCheckTimeoutMsForTests ?? ((_: number | null) => {});
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
  // Paths whose gates ignore abort so a late body can still arrive after switch.
  ignoreGateAbort?: Record<string, true>;
  // Default is a minimal real posture body. Overrides model HTML/fallback and bad JSON.
  postureMode?: "json" | "html" | "empty" | "malformed";
  postureBody?: unknown;
  // Headers OK; body stream errors when the probe signal aborts (body-read timeout).
  postureBodyRejectsOnAbort?: boolean;
  // Headers OK; body stream never completes and ignores abort.
  postureBodyHangIgnoreAbort?: boolean;
  // Transport ignores AbortSignal entirely (does not reject on abort).
  ignoreAbort?: boolean;
  ignoreAbortDelayMs?: number;
  ignoreAbortHang?: boolean;
  // Held /query/stream responses for connection-scoping tests.
  streamGate?: Promise<unknown>;
  streamIgnoreAbort?: boolean;
  streamEvents?: Array<Record<string, unknown>>;
  // Pushable SSE body for cancel / ownership races. When set, /query/stream
  // returns this live stream instead of a one-shot events payload.
  liveStream?: boolean;
  cancelGate?: Promise<unknown>;
  cancelStatus?: number;
  cancelIgnoreAbort?: boolean;
};
type LiveStreamHandle = {
  push: (events: Array<Record<string, unknown>>) => void;
  close: () => void;
  error: (reason: unknown) => void;
};
let liveStreams: LiveStreamHandle[] = [];
const ADMIN_PATHS = ["/config", "/monitoring/traces", "/providers/local"];

const postureFor = (server: Server) =>
  server.postureBody ?? {
    level: "local_only",
    summary: "synthetic",
    settings: {
      auth_enabled: server.key !== null,
      api_keys_configured: server.key !== null,
      exposed_mode: false,
      rate_limit_enabled: true,
      allowed_origin_count: 1,
      wildcard_cors: false,
      docs_public: true,
    },
    checks: [],
    recommendations: [],
  };

const waitGate = async (gate: Promise<unknown> | undefined, signal?: AbortSignal) => {
  if (!gate) return;
  if (!signal) {
    await gate;
    return;
  }
  await new Promise<void>((resolve, reject) => {
    if (signal.aborted) {
      reject(new DOMException("The operation was aborted.", "AbortError"));
      return;
    }
    const onAbort = () => {
      signal.removeEventListener("abort", onAbort);
      reject(new DOMException("The operation was aborted.", "AbortError"));
    };
    signal.addEventListener("abort", onAbort);
    gate.then(
      () => {
        signal.removeEventListener("abort", onAbort);
        resolve();
      },
      err => {
        signal.removeEventListener("abort", onAbort);
        reject(err);
      },
    );
  });
};

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
  if (path === "/security/posture") return postureFor(server);
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

const hangingBody = (signal: AbortSignal | undefined, honorAbort: boolean) => {
  const stream = new ReadableStream<Uint8Array>({
    start(controller) {
      if (!honorAbort) {
        return;
      }
      const boom = () => controller.error(new DOMException("The operation was aborted.", "AbortError"));
      if (signal?.aborted) {
        boom();
        return;
      }
      signal?.addEventListener("abort", boom, { once: true });
    },
  });
  return new Response(stream, { status: 200, headers: { "Content-Type": "application/json" } });
};

const fakeFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
  // A URL without a scheme resolves against the page, like a browser fetch.
  const url = new URL(String(input), location.href);
  const key = new Headers(init?.headers).get("X-API-Key");
  const signal = init?.signal ?? undefined;
  log.push({ origin: url.origin, path: url.pathname, key, signal });
  // The UI server answers every unmatched path with index.html (src/index.ts "/*").
  if (url.origin === UI_ORIGIN) return new Response("<!doctype html>", { status: 200, headers: { "Content-Type": "text/html" } });
  const server = url.origin === ORIGIN_A ? serverA : url.origin === ORIGIN_B ? serverB : undefined;
  if (!server) throw new TypeError("Failed to fetch");

  if (server.ignoreAbort) {
    if (server.ignoreAbortHang) {
      await new Promise(() => {});
    }
    if (server.ignoreAbortDelayMs) {
      await new Promise<void>(resolve => setTimeout(resolve, server.ignoreAbortDelayMs));
    }
  } else {
    if (signal?.aborted && !server.ignoreGateAbort?.[url.pathname] && !(url.pathname === "/query/stream" && server.streamIgnoreAbort)) {
      throw new DOMException("The operation was aborted.", "AbortError");
    }
    if (url.pathname === "/query/stream") {
      await waitGate(server.streamGate, server.streamIgnoreAbort ? undefined : signal);
    } else {
      await waitGate(
        server.gates[url.pathname],
        server.ignoreGateAbort?.[url.pathname] ? undefined : signal,
      );
    }
  }

  if (url.pathname === "/healthz") return json(server.health, { status: server.health < 400 ? "ok" : "error" });
  if (url.pathname === "/security/posture") {
    if (server.postureBodyRejectsOnAbort) {
      return hangingBody(signal, true);
    }
    if (server.postureBodyHangIgnoreAbort) {
      return hangingBody(signal, false);
    }
    if (server.postureMode === "html") {
      return new Response("<!doctype html>", { status: 200, headers: { "Content-Type": "text/html" } });
    }
    if (server.postureMode === "empty") {
      return new Response("", { status: 200, headers: { "Content-Type": "application/json" } });
    }
    if (server.postureMode === "malformed") {
      return new Response("{not-json", { status: 200, headers: { "Content-Type": "application/json" } });
    }
  }
  if (server.key !== null && key === null) return json(401, { detail: "API key required. Provide X-API-Key header." });
  const readOnly = key !== null && key === server.readKey && !ADMIN_PATHS.includes(url.pathname);
  if (server.key !== null && key !== server.key && !readOnly) return json(403, { detail: "Invalid API key or insufficient permissions." });
  if (url.pathname === "/query/cancel") {
    await waitGate(server.cancelGate, server.cancelIgnoreAbort ? undefined : signal);
    return json(server.cancelStatus ?? 200, { ok: true });
  }
  if (url.pathname === "/query/stream") {
    if (server.liveStream) {
      const encoder = new TextEncoder();
      let streamController!: ReadableStreamDefaultController<Uint8Array>;
      const stream = new ReadableStream<Uint8Array>({
        start(controller) {
          streamController = controller;
        },
        cancel() {
          try {
            streamController.close();
          } catch {
            // already closed
          }
        },
      });
      const handle: LiveStreamHandle = {
        push(events) {
          for (const event of events) {
            streamController.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
          }
        },
        close() {
          try {
            streamController.close();
          } catch {
            // already closed
          }
        },
        error(reason) {
          try {
            streamController.error(reason);
          } catch {
            // already closed
          }
        },
      };
      liveStreams.push(handle);
      return new Response(stream, { status: 200, headers: { "Content-Type": "text/event-stream" } });
    }
    const events = server.streamEvents ?? [
      { type: "stage", data: { name: "answering" } },
      {
        type: "result",
        data: {
          answer: "synthetic answer",
          chunks: [],
          sources: [],
          trace_id: "trace-synthetic",
          metrics: {},
          steps: [],
        },
      },
    ];
    const payload = events.map(event => `data: ${JSON.stringify(event)}\n\n`).join("");
    return new Response(payload, { status: 200, headers: { "Content-Type": "text/event-stream" } });
  }
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
  let reject!: (error: unknown) => void;
  const promise = new Promise<void>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
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
  liveStreams = [];
  serverA = { health: 200, key: GOOD_KEY, readKey: null, provider: "Provider A", docs: [], gates: {} };
  serverB = { health: 200, key: GOOD_KEY, readKey: null, provider: "Provider B", docs: [], gates: {} };
  setConnectionCheckTimeoutMsForTests(null);
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
  setConnectionCheckTimeoutMsForTests(null);
  globalThis.fetch = realFetch;
  globalThis.setInterval = realSetInterval;
});

afterAll(async () => {
  await GlobalRegistrator.unregister();
});


const openTab = async (label: string) => {
  const button = [...container.querySelectorAll("button")].find(
    node => node.getAttribute("aria-label") === label || node.textContent?.trim() === label,
  ) as HTMLButtonElement | undefined;
  if (!button) throw new Error(`tab ${label} not found`);
  await act(async () => {
    button.click();
  });
  await settle();
  // ChatInterface is lazy; give the Suspense boundary a couple more turns.
  await settle();
};

const askQuestion = async (question: string) => {
  await openTab("Query");
  const area = container.querySelector('textarea[aria-label="Ask a grounded question"]') as HTMLTextAreaElement;
  if (!area) throw new Error("question textarea missing");
  const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, "value")!.set!;
  await act(async () => {
    setter.call(area, question);
    area.dispatchEvent(new Event("input", { bubbles: true }));
  });
  await settle();
  const ask = container.querySelector('button[aria-label="Ask grounded question"]') as HTMLButtonElement;
  await act(async () => {
    ask.click();
  });
  await settle();
};

const clickCancel = async () => {
  const button = [...container.querySelectorAll("button")].find(node => node.textContent?.trim() === "Cancel") as
    | HTMLButtonElement
    | undefined;
  if (!button) throw new Error("Cancel button not found");
  await act(async () => {
    button.click();
  });
  await settle();
};

const resultEvent = (answer: string, traceId: string) => ({
  type: "result",
  data: {
    answer,
    chunks: [],
    sources: [],
    trace_id: traceId,
    metrics: {},
    steps: [],
  },
});

const progressEvent = (traceId: string) => ({
  type: "progress",
  data: { stage: "answering", message: "working", trace_id: traceId },
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
    // Hold both so the signals under test are still in flight at unmount.
    // checkConnection uses a linked local controller per probe; a completed
    // probe's signal is no longer aborted by the effect cleanup.
    serverA.gates["/config"] = new Promise<void>(() => {});
    serverA.gates["/healthz"] = new Promise<void>(() => {});
    await mount();
    const pending = log.filter(entry => entry.path === "/config");
    const heartbeats = log.filter(entry => entry.path === "/healthz");
    expect(pending.length).toBeGreaterThan(0);
    expect(heartbeats.length).toBeGreaterThan(0);

    await act(async () => root!.unmount());
    root = null;

    expect(pending.every(entry => entry.signal?.aborted === true)).toBe(true);
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

  test("changing to a read-only key at the same URL hides admin config but still loads documents", async () => {
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
    expect(log.some(entry => entry.path === "/documents" && entry.key === READ_KEY)).toBe(true);
    expect(providerShown()).toBe("No provider selected");
    // Read-scoped /documents must not be discarded when admin /config returns 403.
    expect(shown("Documents")).toBe("1");
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

describe("Posture probe contract", () => {
  test("an absolute UI origin HTML 200 is not reported as connected", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.docs = [{ id: "doc-1", title: "Synthetic doc", text: "", metadata: {} }];
    await mount();
    expect(providerShown()).toBe("Provider A");
    expect(shown("Documents")).toBe("1");

    await type("API base URL", UI_ORIGIN);
    await type("API key", "synthetic-ui-key");
    await clickConnect();

    expect(toasts()).toEqual([
      `Connect failed: ${UI_ORIGIN} returned an unexpected response for /security/posture. Still using ${ORIGIN_A}.`,
    ]);
    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider A");
    expect(shown("Documents")).toBe("1");
    expect(sessionStorage.getItem("jr-autorag-api-key")).toBe(GOOD_KEY);
    expect(sessionStorage.getItem("jr-autorag-api-key-origin")).toBe(ORIGIN_A);
  });

  test("HTML 200 from /security/posture is refused and keeps the previous connection", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.docs = [{ id: "doc-1", title: "Synthetic doc", text: "", metadata: {} }];
    await mount();
    serverB.postureMode = "html";

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();

    expect(toasts()).toEqual([
      `Connect failed: ${ORIGIN_B} returned an unexpected response for /security/posture. Still using ${ORIGIN_A}.`,
    ]);
    expect(providerShown()).toBe("Provider A");
    expect(shown("Documents")).toBe("1");
    expect(sessionStorage.getItem("jr-autorag-api-key-origin")).toBe(ORIGIN_A);
  });

  test("malformed JSON from /security/posture is refused", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    serverB.postureMode = "malformed";

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();

    expect(toasts().at(-1)).toBe(
      `Connect failed: ${ORIGIN_B} returned an unexpected response for /security/posture. Still using ${ORIGIN_A}.`,
    );
    expect(providerShown()).toBe("Provider A");
  });

  test("unrelated JSON from /security/posture is refused", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    serverB.postureBody = { status: "ok", healthy: true };

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();

    expect(toasts().at(-1)).toBe(
      `Connect failed: ${ORIGIN_B} returned an unexpected response for /security/posture. Still using ${ORIGIN_A}.`,
    );
    expect(providerShown()).toBe("Provider A");
  });

  test("incomplete posture JSON missing settings.auth_enabled is refused", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    serverB.postureBody = { level: "local_only", summary: "synthetic", settings: {}, checks: [] };

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();

    expect(toasts().at(-1)).toBe(
      `Connect failed: ${ORIGIN_B} returned an unexpected response for /security/posture. Still using ${ORIGIN_A}.`,
    );
    expect(providerShown()).toBe("Provider A");
  });

  test("auth-disabled posture (auth_enabled false) still connects", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    serverB.key = null;

    await type("API base URL", ORIGIN_B);
    await type("API key", "");
    await clickConnect();

    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider B");
    expect(toasts()).toEqual([`Connected: ${ORIGIN_B}`]);
  });

  test("a valid read-only key still connects when posture matches the contract", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.readKey = READ_KEY;
    await mount();

    await type("API key", READ_KEY);
    await clickConnect();

    expect(connectButton().textContent).toBe("Connected");
    expect(sessionStorage.getItem("jr-autorag-api-key")).toBe(READ_KEY);
    expect(log.some(entry => entry.path === "/security/posture" && entry.key === READ_KEY)).toBe(true);
  });
});

describe("Connection check timeout", () => {
  const waitForTimeout = async (ms = 80) => {
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, ms));
    });
    await settle();
  };

  test("a never-resolving /healthz probe times out Connect and keeps the previous connection", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.docs = [{ id: "doc-1", title: "Synthetic doc", text: "", metadata: {} }];
    await mount();
    // Keep the bound longer than clickConnect's settle loop so "Connecting" is
    // observable under slower toolchains (Bun 1.4.0) before the deadline fires.
    setConnectionCheckTimeoutMsForTests(200);
    serverB.gates["/healthz"] = new Promise(() => {});

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    // Still pending before the injectable timeout fires.
    expect(connectButton().textContent).toBe("Connecting");
    await waitForTimeout(280);

    expect(connectButton().textContent).toBe("Connected");
    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_B} timed out. Still using ${ORIGIN_A}.`]);
    expect(providerShown()).toBe("Provider A");
    expect(shown("Documents")).toBe("1");
    expect(log.filter(entry => entry.origin === ORIGIN_B).every(entry => entry.signal?.aborted === true)).toBe(true);
  });

  test("a never-resolving /security/posture probe times out Connect", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    setConnectionCheckTimeoutMsForTests(40);
    serverB.gates["/security/posture"] = new Promise(() => {});

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    await waitForTimeout();

    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_B} timed out. Still using ${ORIGIN_A}.`]);
    expect(providerShown()).toBe("Provider A");
    expect(connectButton().textContent).toBe("Connected");
  });

  test("a never-resolving heartbeat health probe times out and clears Connected", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    expect(connectButton().textContent).toBe("Connected");
    setConnectionCheckTimeoutMsForTests(40);
    serverA.gates["/healthz"] = new Promise(() => {});
    await tickHeartbeat();
    await waitForTimeout();

    expect(connectButton().textContent).toBe("Connect");
    expect(log.filter(entry => entry.path === "/healthz").at(-1)?.signal?.aborted).toBe(true);
  });

  test("editing the draft while a timed Connect is pending cancels it without a timeout toast", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    setConnectionCheckTimeoutMsForTests(200);
    serverB.gates["/healthz"] = new Promise(() => {});

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    expect(connectButton().textContent).toBe("Connecting");
    await type("API key", "synthetic-newer-draft");
    expect(connectButton().textContent).toBe("Connected");
    await waitForTimeout(250);

    expect(toasts()).toEqual([]);
    expect(input("API key").value).toBe("synthetic-newer-draft");
    expect(providerShown()).toBe("Provider A");
  });

  test("unmount aborts a pending Connect timeout and its in-flight fetches", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    setConnectionCheckTimeoutMsForTests(200);
    serverB.gates["/healthz"] = new Promise(() => {});

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    const pending = log.filter(entry => entry.origin === ORIGIN_B);
    expect(pending.length).toBeGreaterThan(0);

    await act(async () => root!.unmount());
    root = null;

    expect(pending.every(entry => entry.signal?.aborted === true)).toBe(true);
  });

  test("a late reply after timeout does not apply the draft connection", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.docs = [{ id: "doc-1", title: "Synthetic doc", text: "", metadata: {} }];
    await mount();
    setConnectionCheckTimeoutMsForTests(40);
    const gate = deferred();
    serverB.gates["/healthz"] = gate.promise;

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    await waitForTimeout();
    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_B} timed out. Still using ${ORIGIN_A}.`]);
    expect(providerShown()).toBe("Provider A");

    await release(gate);
    await settle();

    expect(providerShown()).toBe("Provider A");
    expect(shown("Documents")).toBe("1");
    expect(sessionStorage.getItem("jr-autorag-api-key-origin")).toBe(ORIGIN_A);
    expect(connectButton().textContent).toBe("Connected");
  });
});

describe("Connection check body-stream and abort-ignoring deadline", () => {
  const waitForTimeout = async (ms = 80) => {
    await act(async () => {
      await new Promise(resolve => setTimeout(resolve, ms));
    });
    await settle();
  };

  test("a posture body still reading when the deadline fires is timed out, not unexpected response", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    setConnectionCheckTimeoutMsForTests(20);
    serverB.postureBodyRejectsOnAbort = true;

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    await waitForTimeout(60);

    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_B} timed out. Still using ${ORIGIN_A}.`]);
    expect(providerShown()).toBe("Provider A");
    expect(connectButton().textContent).toBe("Connected");
    expect(sessionStorage.getItem("jr-autorag-api-key-origin")).toBe(ORIGIN_A);
  });

  test("an abort-ignoring transport that answers after the deadline does not connect", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.docs = [{ id: "doc-1", title: "Synthetic doc", text: "", metadata: {} }];
    await mount();
    setConnectionCheckTimeoutMsForTests(20);
    serverB.ignoreAbort = true;
    serverB.ignoreAbortDelayMs = 46;

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    await waitForTimeout(30);
    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_B} timed out. Still using ${ORIGIN_A}.`]);
    expect(providerShown()).toBe("Provider A");

    // Late success arrives after the operation settled; must not apply the draft.
    await waitForTimeout(40);
    await settle();

    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_B} timed out. Still using ${ORIGIN_A}.`]);
    expect(providerShown()).toBe("Provider A");
    expect(shown("Documents")).toBe("1");
    expect(sessionStorage.getItem("jr-autorag-api-key-origin")).toBe(ORIGIN_A);
    expect(connectButton().textContent).toBe("Connected");
  });

  test("an abort-ignoring never-resolving fetch settles as timed out instead of hanging", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    setConnectionCheckTimeoutMsForTests(20);
    serverB.ignoreAbort = true;
    serverB.ignoreAbortHang = true;

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    await waitForTimeout(60);

    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_B} timed out. Still using ${ORIGIN_A}.`]);
    expect(providerShown()).toBe("Provider A");
    expect(connectButton().textContent).toBe("Connected");
  });

  test("an abort-ignoring hanging posture body settles as timed out", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    setConnectionCheckTimeoutMsForTests(20);
    serverB.postureBodyHangIgnoreAbort = true;

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    await waitForTimeout(60);

    expect(toasts()).toEqual([`Connect failed: ${ORIGIN_B} timed out. Still using ${ORIGIN_A}.`]);
    expect(providerShown()).toBe("Provider A");
    expect(connectButton().textContent).toBe("Connected");
  });

  test("a late abort-ignoring heartbeat success after timeout does not restore Connected", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    expect(connectButton().textContent).toBe("Connected");
    setConnectionCheckTimeoutMsForTests(20);
    serverA.ignoreAbort = true;
    serverA.ignoreAbortDelayMs = 46;

    await tickHeartbeat();
    await waitForTimeout(30);
    expect(connectButton().textContent).toBe("Connect");

    await waitForTimeout(40);
    await settle();
    expect(connectButton().textContent).toBe("Connect");
  });
});

describe("Status ownership and permission-scoped refresh", () => {
  test("a late successful query stream after a failed heartbeat does not restore Connected", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    expect(connectButton().textContent).toBe("Connected");

    serverA.health = 503;
    await tickHeartbeat();
    expect(connectButton().textContent).toBe("Connect");

    const stream = deferred();
    serverA.streamGate = stream.promise;
    await askQuestion("synthetic late stream question");
    expect(log.some(entry => entry.path === "/query/stream")).toBe(true);
    expect(connectButton().textContent).toBe("Connect");

    await release(stream);
    expect(connectButton().textContent).toBe("Connect");
    expect(container.textContent).toContain("synthetic answer");
  });

  test("an old query stream after a connection switch does not restore status or leak into the new connection", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.docs = [{ id: "doc-a", title: "Doc A", text: "", metadata: {} }];
    serverB.docs = [{ id: "doc-b", title: "Doc B", text: "", metadata: {} }];
    await mount();
    expect(shown("Documents")).toBe("1");

    const stream = deferred();
    serverA.streamGate = stream.promise;
    serverA.streamIgnoreAbort = true;
    await askQuestion("synthetic old-stream question");

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider B");
    expect(shown("Documents")).toBe("1");

    serverA.streamEvents = [
      { type: "stage", data: { name: "answering" } },
      {
        type: "result",
        data: {
          answer: "stale answer from origin A",
          chunks: [],
          sources: [],
          trace_id: "trace-stale",
          metrics: {},
          steps: [],
        },
      },
    ];
    await release(stream);
    await settle();

    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider B");
    expect(shown("Documents")).toBe("1");
    expect(container.textContent).not.toContain("stale answer from origin A");
    // refreshAll from the stale stream must not push origin A admin config back.
    expect(providerShown()).not.toBe("Provider A");
    expect(toasts().some(t => t.includes("Query interrupted"))).toBe(false);
    expect(container.textContent).not.toMatch(/Query failed:/);
  });

  test("admin-to-read-only switch loads documents even when admin endpoints are forbidden", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.readKey = READ_KEY;
    serverA.docs = [
      { id: "doc-1", title: "Readable doc", text: "", metadata: {} },
      { id: "doc-2", title: "Second readable doc", text: "", metadata: {} },
    ];
    await mount();
    expect(providerShown()).toBe("Provider A");
    expect(shown("Documents")).toBe("2");

    await type("API key", READ_KEY);
    await clickConnect();

    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("No provider selected");
    expect(shown("Documents")).toBe("2");
    expect(log.filter(entry => entry.path === "/config" && entry.key === READ_KEY).length).toBeGreaterThan(0);
    expect(log.filter(entry => entry.path === "/monitoring/traces" && entry.key === READ_KEY).length).toBeGreaterThan(0);
    expect(log.filter(entry => entry.path === "/documents" && entry.key === READ_KEY).length).toBeGreaterThan(0);
  });

  test("a stale admin config reply after switching to read-only cannot reappear", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.readKey = READ_KEY;
    serverA.docs = [{ id: "doc-1", title: "Readable doc", text: "", metadata: {} }];
    const staleConfig = deferred();
    serverA.gates["/config"] = staleConfig.promise;
    serverA.ignoreGateAbort = { "/config": true };
    await mount();
    // Documents settle independently while admin config is held.
    expect(shown("Documents")).toBe("1");
    expect(providerShown()).toBe("No provider selected");

    await type("API key", READ_KEY);
    await clickConnect();
    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("No provider selected");
    expect(shown("Documents")).toBe("1");

    serverA.provider = "Stale Admin Provider";
    await release(staleConfig);
    await settle();

    expect(providerShown()).toBe("No provider selected");
    expect(shown("Documents")).toBe("1");
    expect(connectButton().textContent).toBe("Connected");
  });

  test("ordinary heartbeat recovery still returns to Connected", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    expect(connectButton().textContent).toBe("Connected");

    serverA.health = 503;
    await tickHeartbeat();
    expect(connectButton().textContent).toBe("Connect");

    serverA.health = 200;
    await tickHeartbeat();
    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider A");
  });
});

describe("Query ownership", () => {
  test("an old stream finalization after a newer same-connection query starts does not overwrite it", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    expect(connectButton().textContent).toBe("Connected");

    const first = deferred();
    serverA.streamGate = first.promise;
    serverA.streamIgnoreAbort = true;
    await askQuestion("first ownership question");
    expect(log.some(entry => entry.path === "/query/stream")).toBe(true);

    await clickCancel();
    expect(container.querySelector('button[aria-label="Ask grounded question"]')).toBeTruthy();

    serverA.streamGate = undefined;
    serverA.streamIgnoreAbort = false;
    serverA.streamEvents = [
      { type: "stage", data: { name: "answering" } },
      resultEvent("second query answer", "trace-second"),
    ];
    await askQuestion("second ownership question");
    await settle();
    expect(container.textContent).toContain("second query answer");

    serverA.streamEvents = [
      { type: "stage", data: { name: "answering" } },
      resultEvent("stale first query answer", "trace-first"),
    ];
    await release(first);
    await settle();

    expect(container.textContent).toContain("second query answer");
    expect(container.textContent).not.toContain("stale first query answer");
    expect(toasts().some(t => t.includes("Query interrupted"))).toBe(false);
    expect(container.querySelector('button[aria-label="Query running"]')).toBeNull();
  });

  test("cancel then an immediate same-connection query keeps the new stream", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.liveStream = true;
    await mount();

    await askQuestion("live cancel question");
    expect(liveStreams.length).toBe(1);
    liveStreams[0]!.push([progressEvent("trace-live-1"), { type: "stage", data: { name: "answering" } }]);
    await settle();
    expect(container.querySelector("button") && [...container.querySelectorAll("button")].some(b => b.textContent?.trim() === "Cancel")).toBe(true);

    await clickCancel();
    expect(log.some(entry => entry.path === "/query/cancel" && entry.origin === ORIGIN_A)).toBe(true);

    serverA.liveStream = false;
    serverA.streamEvents = [
      { type: "stage", data: { name: "answering" } },
      resultEvent("post-cancel answer", "trace-live-2"),
    ];
    await askQuestion("after cancel question");
    await settle();

    expect(container.textContent).toContain("post-cancel answer");
    expect(toasts().some(t => t.includes("Query cancelled"))).toBe(true);
    liveStreams[0]!.push([resultEvent("cancelled stream answer", "trace-live-1")]);
    liveStreams[0]!.close();
    await settle();
    expect(container.textContent).not.toContain("cancelled stream answer");
    expect(container.textContent).toContain("post-cancel answer");
  });

  test("a late cancel response does not clear a newer query", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.liveStream = true;
    const cancelHold = deferred();
    serverA.cancelGate = cancelHold.promise;
    serverA.cancelIgnoreAbort = true;
    await mount();

    await askQuestion("late cancel first");
    liveStreams[0]!.push([progressEvent("trace-late-1")]);
    await settle();

    await clickCancel();
    expect(log.some(entry => entry.path === "/query/cancel")).toBe(true);

    serverA.liveStream = false;
    serverA.streamEvents = [
      { type: "stage", data: { name: "answering" } },
      resultEvent("newer query after late cancel", "trace-late-2"),
    ];
    await askQuestion("late cancel second");
    await settle();
    expect(container.textContent).toContain("newer query after late cancel");
    expect(container.querySelector('button[aria-label="Query running"]')).toBeNull();

    // Begin a third in-flight query so a late cancel clear would be visible.
    serverA.liveStream = true;
    await askQuestion("late cancel third in flight");
    expect(liveStreams.length).toBeGreaterThanOrEqual(2);
    liveStreams[liveStreams.length - 1]!.push([progressEvent("trace-late-3"), { type: "stage", data: { name: "gatherer" } }]);
    await settle();
    expect(container.querySelector('button[aria-label="Query running"]')).toBeTruthy();
    expect(container.textContent).toMatch(/gatherer|Running|Streaming/i);

    await release(cancelHold);
    await settle();

    expect(container.querySelector('button[aria-label="Query running"]')).toBeTruthy();
    expect(container.textContent).toMatch(/gatherer|Running|Streaming/i);
  });

  test("cancel HTTP failure does not announce cancellation or clear a newer query", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.liveStream = true;
    const cancelHold = deferred();
    serverA.cancelGate = cancelHold.promise;
    serverA.cancelIgnoreAbort = true;
    serverA.cancelStatus = 500;
    await mount();

    await askQuestion("cancel failure first");
    liveStreams[0]!.push([progressEvent("trace-fail-1")]);
    await settle();
    await clickCancel();

    serverA.liveStream = false;
    serverA.streamEvents = [
      { type: "stage", data: { name: "answering" } },
      resultEvent("survives cancel failure", "trace-fail-2"),
    ];
    await askQuestion("cancel failure second");
    await settle();
    expect(container.textContent).toContain("survives cancel failure");

    await release(cancelHold);
    await settle();

    expect(toasts().some(t => t.includes("Query cancelled"))).toBe(false);
    expect(container.textContent).toContain("survives cancel failure");
    expect(toasts().some(t => t.includes("Cancel failed"))).toBe(false);
  });

  test("old stream rejection after URL/key switch does not toast Query interrupted", async () => {
    storeSessionKey(GOOD_KEY);
    await mount();
    const stream = deferred();
    serverA.streamGate = stream.promise;
    serverA.streamIgnoreAbort = true;
    await askQuestion("switch reject question");

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    expect(connectButton().textContent).toBe("Connected");

    // The replaced stream's fetch rejects (network failure), not a late success.
    await act(async () => {
      stream.reject(new TypeError("Failed to fetch"));
    });
    await settle();

    expect(toasts().some(t => t.includes("Query interrupted"))).toBe(false);
    expect(container.textContent).not.toContain("Failed to fetch");
    expect(container.textContent).not.toMatch(/Query failed:/);
    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider B");
  });

  test("an old stream reader error after URL/key switch does not toast Query interrupted", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.liveStream = true;
    await mount();
    await askQuestion("switch reader error question");
    expect(liveStreams.length).toBe(1);
    liveStreams[0]!.push([{ type: "stage", data: { name: "answering" } }]);
    await settle();

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    expect(connectButton().textContent).toBe("Connected");

    liveStreams[0]!.error(new TypeError("network error while reading"));
    await settle();

    expect(toasts().some(t => t.includes("Query interrupted"))).toBe(false);
    expect(container.textContent).not.toContain("network error while reading");
    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider B");
  });
});

describe("Cancel across a connection switch", () => {
  // Ask on A, get a trace id, press Cancel while the cancel POST is held, then
  // apply B with no newer Ask. The held reply belongs to A and must stay silent.
  const cancelThenSwitch = async (cancelGate: Promise<unknown>) => {
    storeSessionKey(GOOD_KEY);
    serverA.liveStream = true;
    serverA.cancelGate = cancelGate;
    serverA.cancelIgnoreAbort = true;
    await mount();
    await askQuestion("cancel then switch");
    liveStreams[0]!.push([progressEvent("trace-switch-cancel")]);
    await settle();
    await clickCancel();
    expect(log.some(entry => entry.path === "/query/cancel" && entry.origin === ORIGIN_A)).toBe(true);

    await type("API base URL", ORIGIN_B);
    await type("API key", GOOD_KEY);
    await clickConnect();
    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider B");
  };

  const expectCurrentConnectionUntouched = () => {
    expect(toasts().some(t => /Query cancelled|Cancel failed|Stopped streaming/.test(t))).toBe(false);
    expect(container.textContent).not.toMatch(/Query cancelled|Cancel failed/);
    expect(connectButton().textContent).toBe("Connected");
    expect(providerShown()).toBe("Provider B");
  };

  test("a delayed cancel success after URL/key switch does not announce cancellation", async () => {
    const hold = deferred();
    await cancelThenSwitch(hold.promise);
    const toastsBefore = toasts().length;
    await release(hold);
    await settle();
    expectCurrentConnectionUntouched();
    expect(toasts().length).toBe(toastsBefore);
  });

  test("a delayed cancel HTTP failure after URL/key switch stays silent", async () => {
    const hold = deferred();
    serverA.cancelStatus = 500;
    await cancelThenSwitch(hold.promise);
    const toastsBefore = toasts().length;
    await release(hold);
    await settle();
    expectCurrentConnectionUntouched();
    expect(toasts().length).toBe(toastsBefore);
  });

  test("a delayed cancel network failure after URL/key switch stays silent", async () => {
    const hold = deferred();
    await cancelThenSwitch(hold.promise);
    const toastsBefore = toasts().length;
    await act(async () => {
      hold.reject(new TypeError("Failed to fetch"));
    });
    await settle();
    expectCurrentConnectionUntouched();
    expect(toasts().length).toBe(toastsBefore);
  });

  test("a cancel HTTP failure with no replacement tells the user it failed", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.liveStream = true;
    serverA.cancelStatus = 500;
    await mount();
    await askQuestion("cancel fails in place");
    liveStreams[0]!.push([progressEvent("trace-cancel-fails")]);
    await settle();
    await clickCancel();
    await settle();

    const failed = toasts().find(t => t.includes("Cancel failed"));
    expect(failed).toBeTruthy();
    expect(failed).not.toBe("Cancel failed");
    expect(toasts().some(t => t.includes("Query cancelled"))).toBe(false);
  });

  test("cancel before the server reports a trace says only local streaming stopped", async () => {
    storeSessionKey(GOOD_KEY);
    serverA.liveStream = true;
    await mount();
    await askQuestion("cancel before trace");
    liveStreams[0]!.push([{ type: "stage", data: { name: "answering" } }]);
    await settle();
    await clickCancel();

    expect(log.some(entry => entry.path === "/query/cancel")).toBe(false);
    expect(toasts().some(t => t.includes("Stopped streaming") && t.includes("not asked to cancel"))).toBe(true);
    expect(toasts().some(t => t.includes("Query cancelled"))).toBe(false);
  });
});

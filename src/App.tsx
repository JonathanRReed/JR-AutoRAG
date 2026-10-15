import { ScopedActivity } from "./lib/scoped-activity";
import { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Database,
  FileCheck2,
  FileText,
  Home,
  MessageSquare,
  Moon,
  PackageCheck,
  Settings,
  ShieldCheck,
  Sun,
  TerminalSquare,
  Trophy,
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/toast";
import { LoadingSpinner } from "@/components/ui/loading";
import { buildApiUrl, normalizeApiBaseUrl, resolveDefaultApiBaseUrl } from "@/lib/api-url";

const AdvancedRAGSettings = lazy(() => import("@/components/features/AdvancedRAGSettings").then(m => ({ default: m.AdvancedRAGSettings })));
const ChatInterface = lazy(() => import("@/components/features/ChatInterface").then(m => ({ default: m.ChatInterface })));
const EnterpriseStatusPanel = lazy(() => import("@/components/features/EnterpriseStatusPanel").then(m => ({ default: m.EnterpriseStatusPanel })));
const IngestPanel = lazy(() => import("@/components/features/IngestPanel").then(m => ({ default: m.IngestPanel })));
const MetricsDashboard = lazy(() => import("@/components/features/MetricsDashboard").then(m => ({ default: m.MetricsDashboard })));
const OnboardingFlow = lazy(() => import("@/components/features/OnboardingFlow").then(m => ({ default: m.OnboardingFlow })));
const ProviderConfig = lazy(() => import("@/components/features/ProviderConfig").then(m => ({ default: m.ProviderConfig })));
const ProviderCarousel = lazy(() => import("@/components/features/ProviderCarousel").then(m => ({ default: m.ProviderCarousel })));
const QualityCockpit = lazy(() => import("@/components/features/QualityCockpit").then(m => ({ default: m.QualityCockpit })));
const TraceLog = lazy(() => import("@/components/features/TraceLog").then(m => ({ default: m.TraceLog })));
const PresetSelector = lazy(() => import("@/components/features/PresetSelector").then(m => ({ default: m.PresetSelector })));

import "./index.css";
import type {
  AppConfig,
  CacheStats,
  DocumentOut,
  IngestResponse,
  InstallReportResponse,
  LocalProviderInfo,
  ModelStatus,
  ProviderConfig as ProviderConfigType,
  ProviderProfile,
  QueryResponse,
  UploadBatch,
  RetrievalDefaults,
  RoleSelection,
  SecurityPostureResponse,
  TraceOut,
  ChatSession,
  PresetLevel,
  QueryMode,
  SubsystemBackendConfig,
  OCRPolicy,
  DemoSeedResponse,
  OnboardingState,
} from "@/types";

const defaultBaseUrl = resolveDefaultApiBaseUrl();

const formatNumber = (value?: number) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "0.00";

const toMessage = (error: unknown) => {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
};

const originOf = (value: string) => {
  try {
    return new URL(normalizeApiBaseUrl(value)).origin;
  } catch {
    return null;
  }
};

type ConnectionCheck = { ok: true } | { ok: false; reason: string };

// Bound each heartbeat/Connect probe so a hung host cannot spin Connect forever.
// Deadline settles the caller-facing promise even when fetch/body ignore abort.
export const CONNECTION_CHECK_TIMEOUT_MS = 10_000;

export type ConnectionProbeTimeout = {
  timeoutMs: number;
};

const defaultConnectionProbeTimeout: ConnectionProbeTimeout = {
  timeoutMs: CONNECTION_CHECK_TIMEOUT_MS,
};

let connectionProbeTimeout: ConnectionProbeTimeout = defaultConnectionProbeTimeout;

// Injectable timeout for tests. Prefer this over exporting mutable settle/epoch flags.
export const setConnectionProbeTimeoutForTests = (timeout: ConnectionProbeTimeout | null) => {
  connectionProbeTimeout = timeout == null ? defaultConnectionProbeTimeout : timeout;
};

export const setConnectionCheckTimeoutMsForTests = (ms: number | null) => {
  setConnectionProbeTimeoutForTests(ms == null ? null : { timeoutMs: ms });
};

// Status text is not rendered; tests observe every write through this hook.
let statusObserverForTests: ((status: string) => void) | null = null;
export const observeStatusForTests = (observer: ((status: string) => void) | null) => {
  statusObserverForTests = observer;
};

// Operation identity. Every operation captures the applied connection once, when
// it starts, and uses that same immutable value for its transport (URL + key) and
// for deciding whether any later state write, toast, refresh, or poll may still
// happen. Connect to another URL/key replaces the value; retrying the applied
// URL/key keeps it. Nothing defaults to "whatever is current now" after an await,
// so an old handler can never pair its own transport with a newer generation.
export type ConnectionScope = Readonly<{
  generation: number;
  baseUrl: string;
  apiKey: string;
}>;

// Thrown by scoped polling once its connection has been replaced.
export class StaleConnectionError extends Error {
  constructor() {
    super("Connection changed before this operation finished");
    this.name = "StaleConnectionError";
  }
}

const JSON_HEADERS: HeadersInit = { "Content-Type": "application/json" };

const scopedHeaders = (scope: ConnectionScope, extra?: HeadersInit) => {
  const merged = new Headers(extra);
  // The scope's key is the only key this operation may send.
  merged.delete("X-API-Key");
  const trimmedKey = scope.apiKey.trim();
  if (trimmedKey) {
    merged.set("X-API-Key", trimmedKey);
  }
  return merged;
};

const scopedFetch = (scope: ConnectionScope, path: string, init?: RequestInit) =>
  fetch(buildApiUrl(scope.baseUrl, path), { ...init, headers: scopedHeaders(scope, init?.headers) });

// Connection status is owned by checkConnection (heartbeat and Connect). Data
// requests never write it, so a late response from a replaced connection can't.
const scopedFetchJson = async <T,>(scope: ConnectionScope, path: string, init?: RequestInit): Promise<T> => {
  const response = await scopedFetch(scope, path, init);
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
};

const isHttpUrl = (value: string) => {
  try {
    const { protocol } = new URL(value);
    return protocol === "http:" || protocol === "https:";
  } catch {
    return false;
  }
};

// Minimal fields agreed by api/app/schemas/security.py SecurityPostureResponse
// and src/types.ts SecurityPostureResponse. Present with auth on or off.
const SECURITY_POSTURE_LEVELS = new Set([
  "local_only",
  "client_ready",
  "needs_attention",
  "unsafe",
]);

const isSecurityPostureProbeBody = (value: unknown): boolean => {
  if (!value || typeof value !== "object") {
    return false;
  }
  const body = value as Record<string, unknown>;
  if (typeof body.level !== "string" || !SECURITY_POSTURE_LEVELS.has(body.level)) {
    return false;
  }
  if (typeof body.summary !== "string") {
    return false;
  }
  const settings = body.settings;
  if (!settings || typeof settings !== "object") {
    return false;
  }
  return typeof (settings as Record<string, unknown>).auth_enabled === "boolean";
};

// Fetch + posture body parse only. AbortError from fetch/body propagates so the
// bounded wrapper can classify timeout vs external cancel. JSON parse failures
// stay "unexpected response" and are not confused with abort.
const runConnectionProbe = async (
  baseUrl: string,
  apiKey: string,
  signal: AbortSignal,
): Promise<ConnectionCheck> => {
  const health = await fetch(buildApiUrl(baseUrl, "/healthz"), { method: "GET", signal });
  if (!health.ok) {
    return { ok: false, reason: `returned HTTP ${health.status} for /healthz` };
  }
  const key = apiKey.trim();
  const probe = await fetch(buildApiUrl(baseUrl, "/security/posture"), {
    method: "GET",
    headers: key ? { "X-API-Key": key } : undefined,
    signal,
  });
  if (probe.status === 401 || probe.status === 403) {
    return { ok: false, reason: `rejected the API key (HTTP ${probe.status})` };
  }
  if (!probe.ok) {
    return { ok: false, reason: `returned HTTP ${probe.status} for /security/posture` };
  }
  const text = await probe.text();
  let body: unknown;
  try {
    body = JSON.parse(text);
  } catch {
    return { ok: false, reason: "returned an unexpected response for /security/posture" };
  }
  if (!isSecurityPostureProbeBody(body)) {
    return { ok: false, reason: "returned an unexpected response for /security/posture" };
  }
  return { ok: true };
};

// Shared by heartbeat and Connect. Settles at the deadline independently of
// whether fetch or the response body honor abort, while still aborting the
// transport and cleaning the timer + external listener. A late probe result
// cannot resolve again after the operation has settled.
const checkConnection = (
  baseUrl: string,
  apiKey: string,
  signal: AbortSignal,
): Promise<ConnectionCheck> => {
  // Without a scheme, fetch resolves the URL against the page, and the UI
  // server answers every unknown path with index.html and HTTP 200.
  if (!isHttpUrl(baseUrl)) {
    return Promise.resolve({ ok: false, reason: "is not an http:// or https:// URL" });
  }
  if (signal.aborted) {
    return Promise.resolve({ ok: false, reason: "is unreachable" });
  }

  const local = new AbortController();
  let settled = false;

  return new Promise<ConnectionCheck>(resolve => {
    let timer: ReturnType<typeof setTimeout> | undefined;

    const finish = (result: ConnectionCheck) => {
      if (settled) {
        return;
      }
      settled = true;
      if (timer !== undefined) {
        clearTimeout(timer);
      }
      signal.removeEventListener("abort", onExternalAbort);
      resolve(result);
    };

    const onExternalAbort = () => {
      local.abort();
      finish({ ok: false, reason: "is unreachable" });
    };
    signal.addEventListener("abort", onExternalAbort);

    timer = setTimeout(() => {
      local.abort();
      finish({ ok: false, reason: "timed out" });
    }, connectionProbeTimeout.timeoutMs);

    runConnectionProbe(baseUrl, apiKey, local.signal).then(
      result => {
        if (signal.aborted) {
          finish({ ok: false, reason: "is unreachable" });
          return;
        }
        if (local.signal.aborted) {
          finish({ ok: false, reason: "timed out" });
          return;
        }
        finish(result);
      },
      () => {
        if (signal.aborted) {
          finish({ ok: false, reason: "is unreachable" });
          return;
        }
        if (local.signal.aborted) {
          finish({ ok: false, reason: "timed out" });
          return;
        }
        finish({ ok: false, reason: "is unreachable" });
      },
    );
  });
};

const readStoredJson = <T,>(key: string, fallback: T): T => {
  if (typeof window === "undefined") {
    return fallback;
  }
  const saved = localStorage.getItem(key);
  if (!saved) {
    return fallback;
  }
  try {
    return JSON.parse(saved) as T;
  } catch {
    localStorage.removeItem(key);
    return fallback;
  }
};

const formatDateTime = (value?: string) => {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }
  return date.toLocaleString();
};

type TabId = "home" | "config" | "documents" | "query" | "quality" | "metrics";

const tabs: { id: TabId; label: string; icon: typeof Settings }[] = [
  { id: "home", label: "Home", icon: Home },
  { id: "config", label: "Configuration", icon: Settings },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "query", label: "Query", icon: MessageSquare },
  { id: "quality", label: "Quality", icon: Trophy },
  { id: "metrics", label: "Metrics", icon: BarChart3 },
];

const validTabs = tabs.map(tab => tab.id);

export function App() {
  const { toast } = useToast();
  // Applied connection: every request uses these. The inputs edit the draft
  // values, which are applied only when the user clicks Connect.
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [apiKey, setApiKey] = useState(() => {
    if (typeof window === "undefined") {
      return "";
    }
    // Only reuse a session key that was saved for the origin we start on.
    if (sessionStorage.getItem("jr-autorag-api-key-origin") !== originOf(defaultBaseUrl)) {
      return "";
    }
    return sessionStorage.getItem("jr-autorag-api-key") || "";
  });
  const [draftBaseUrl, setDraftBaseUrl] = useState(baseUrl);
  const [draftApiKey, setDraftApiKey] = useState(apiKey);
  const [connectAttempt, setConnectAttempt] = useState(0);
  const pendingConnect = useRef<AbortController | null>(null);
  // Numbers each status probe of the applied connection (heartbeat tick or a
  // Connect retry). Only the latest one started may write isConnected.
  const latestStatusProbe = useRef(0);
  // The applied connection as one immutable value (see ConnectionScope). Connect
  // replaces it when the URL/key change. Operations capture it at their start.
  const appliedConnection = useRef<ConnectionScope>(Object.freeze({ generation: 0, baseUrl, apiKey }));
  const [activities] = useState(() => new ScopedActivity());
  const currentScope = () => appliedConnection.current;
  const isCurrentScope = (scope: ConnectionScope) => scope === appliedConnection.current;
  const [status, setStatusState] = useState("");
  const setStatus = (value: string) => {
    statusObserverForTests?.(value);
    setStatusState(value);
  };
  const [isConnected, setIsConnected] = useState(false); // New state for reliable connection tracking
  const [isTestingConnection, setIsTestingConnection] = useState(false);
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [traces, setTraces] = useState<TraceOut[]>([]);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [securityPosture, setSecurityPosture] = useState<SecurityPostureResponse | null>(null);
  const [question, setQuestion] = useState("");
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [chatHistory, setChatHistory] = useState<{ role: string; content: string }[]>(() => {
    return readStoredJson("chatHistory", []);
  });
  const [savedSessions, setSavedSessions] = useState<ChatSession[]>(() => {
    return readStoredJson("savedSessions", []);
  });
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const currentSessionIdRef = useRef(currentSessionId);
  useEffect(() => {
    currentSessionIdRef.current = currentSessionId;
  }, [currentSessionId]);
  const [currentTraceId, setCurrentTraceId] = useState<string | null>(null);
  // Each Ask gets a query id. Cancel and cleanup compare that id, not only the
  // connection, so a cancelled query cannot clear a newer one on the same
  // URL/key. The stream's scope keeps origin/key bound for cancel POSTs.
  type ActiveQueryTransport = {
    queryId: number;
    scope: ConnectionScope;
    controller: AbortController;
  };
  const queryIdSeq = useRef(0);
  const activeQueryTransport = useRef<ActiveQueryTransport | null>(null);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [ingestTitle, setIngestTitle] = useState("Getting Started");
  const [ingestText, setIngestText] = useState("Paste onboarding doc content here...");
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [activeStage, setActiveStage] = useState<string | null>(null);
  const [progress, setProgress] = useState<{
    stage: string;
    message: string;
    detail?: string;
    progress?: number;
    elapsed_ms?: number;
    estimated_remaining_ms?: number;
  } | null>(null);
  const discardActiveQuery = () => {
    const active = activeQueryTransport.current;
    if (!active) return null;
    activeQueryTransport.current = null;
    active.controller.abort();
    setIsQuerying(false);
    setActiveStage(null);
    setProgress(null);
    setCurrentTraceId(null);
    return active;
  };
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationSummary, setEvaluationSummary] = useState("");
  const [selectedProfile, setSelectedProfile] = useState("Default");
  const [newProfileName, setNewProfileName] = useState("Default");
  const [modelOptions, setModelOptions] = useState<string[]>([]);
  const [localProviders, setLocalProviders] = useState<LocalProviderInfo[]>([]);
  const [localProvidersStatus, setLocalProvidersStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [localSelections, setLocalSelections] = useState<Record<string, RoleSelection>>({});
  const [isUploadingFile, setIsUploadingFile] = useState(false);
  const [isClearingCache, setIsClearingCache] = useState(false);
  const [onboarding, setOnboarding] = useState<OnboardingState | null>(null);
  const [isSeedingDemo, setIsSeedingDemo] = useState(false);
  const [onboardingMode, setOnboardingMode] = useState<"guided" | "advanced">("guided");
  const [ingestSync, setIngestSync] = useState(true);
  const [langextractProfileOverride, setLangextractProfileOverride] = useState("__global__");
  const [langextractPromptOverride, setLangextractPromptOverride] = useState("");
  const [activeTab, setActiveTab] = useState<TabId>(() => {
    if (typeof window === "undefined") return "home";
    const saved = localStorage.getItem("activeTab");
    if (saved && validTabs.includes(saved as TabId)) {
      return saved as TabId;
    }
    return "home";
  });

  useEffect(() => {
    localStorage.setItem("activeTab", activeTab);
  }, [activeTab]);
  const [modelStatus, setModelStatus] = useState<ModelStatus>({
    embedding: "unknown",
    reranker: "unknown",
  });
  const [checklistDismissed, setChecklistDismissed] = useState(false);
  const [isCheckingModels, setIsCheckingModels] = useState(false);
  const [isExportingInstallReport, setIsExportingInstallReport] = useState(false);
  const [isDownloadingEmbedding, setIsDownloadingEmbedding] = useState(false);
  const [isDownloadingReranker, setIsDownloadingReranker] = useState(false);
  const [modelActionMessage, setModelActionMessage] = useState("");
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window !== "undefined") {
      return localStorage.getItem("jr-autorag-theme") !== "light";
    }
    return true;
  });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const buildHeaders = useCallback(
    (extra?: HeadersInit) => {
      const merged = new Headers(extra);
      const trimmedKey = apiKey.trim();
      if (trimmedKey) {
        merged.set("X-API-Key", trimmedKey);
      }
      return merged;
    },
    [apiKey],
  );
  const [activePreset, setActivePreset] = useState<PresetLevel>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("jr-autorag-preset") as PresetLevel | null;
      if (saved && ["turbo", "fast", "balanced", "thorough", "ultra_accurate"].includes(saved)) {
        return saved;
      }
    }
    return "balanced";
  });
  const [queryMode, setQueryMode] = useState<QueryMode>("grounded");

  // Toggle dark mode
  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(chatHistory));
  }, [chatHistory]);

  useEffect(() => {
    localStorage.setItem("savedSessions", JSON.stringify(savedSessions));
  }, [savedSessions]);

  useEffect(() => {
    localStorage.setItem("jr-autorag-preset", activePreset);
  }, [activePreset]);

  useEffect(() => {
    if (config?.query_mode) {
      setQueryMode(config.query_mode);
    }
  }, [config?.query_mode]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const trimmed = apiKey.trim();
    const origin = originOf(baseUrl);
    if (trimmed && origin) {
      sessionStorage.setItem("jr-autorag-api-key", trimmed);
      sessionStorage.setItem("jr-autorag-api-key-origin", origin);
    } else {
      sessionStorage.removeItem("jr-autorag-api-key");
      sessionStorage.removeItem("jr-autorag-api-key-origin");
    }
  }, [apiKey, baseUrl]);

  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("jr-autorag-theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("jr-autorag-theme", "light");
    }
  }, [isDarkMode]);

  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (activeTab === "query") {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatHistory, queryResult?.answer, activeTab]);

  // Only for child components that issue their own requests (QualityCockpit).
  // App operations use scopedFetch/scopedFetchJson with a captured scope.
  const buildUrl = (path: string) => buildApiUrl(baseUrl, path);
  const fetchJson = scopedFetchJson;

  // Heartbeat for connection status
  useEffect(() => {
    const controller = new AbortController();
    const check = async () => {
      const probe = ++latestStatusProbe.current;
      const result = await checkConnection(baseUrl, apiKey, controller.signal);
      if (!controller.signal.aborted && probe === latestStatusProbe.current) {
        setIsConnected(result.ok);
      }
    };

    // Initial check
    check();

    // Periodically check every 20 seconds
    const id = setInterval(check, 20000);
    return () => {
      controller.abort();
      clearInterval(id);
    };
  }, [baseUrl, apiKey]);

  useEffect(() => () => pendingConnect.current?.abort(), []);

  const refreshLocalProviders = async (scope: ConnectionScope, signal?: AbortSignal) => {
    const owns = () => !signal?.aborted && isCurrentScope(scope);
    if (!owns()) return;
    setLocalProvidersStatus("loading");
    try {
      const providers = await fetchJson<LocalProviderInfo[]>(scope, "/providers/local", { signal });
      if (!owns()) return;
      setLocalProviders(providers);
      setLocalProvidersStatus("ready");
      if (!providers.length) {
        setStatus("No local runtimes detected. Launch Ollama or LM Studio and rescan.");
      }
    } catch (error) {
      if (!owns()) return;
      setLocalProvidersStatus("error");
      setStatus(`Local provider scan failed: ${toMessage(error)}`);
    }
  };

  // Polls on the scope the upload/ingest started on, and stops (throwing
  // StaleConnectionError) as soon as that connection is replaced.
  const waitForDocumentReady = async (scope: ConnectionScope, documentId: string, label?: string) => {
    const timeoutMs = 120_000;
    const intervalMs = 1500;
    const deadline = Date.now() + timeoutMs;
    let lastDocs: DocumentOut[] = [];
    const ensureCurrent = () => {
      if (!isCurrentScope(scope)) throw new StaleConnectionError();
    };

    while (Date.now() < deadline) {
      ensureCurrent();
      const docs = await fetchJson<DocumentOut[]>(scope, "/documents");
      ensureCurrent();
      lastDocs = docs;
      const doc = docs.find(item => item.id === documentId);
      if (doc) {
        const processingStatus = doc.metadata?.processing_status;
        if (!processingStatus || processingStatus === "ready") {
          setDocuments(docs);
          if (label) {
            setStatus(`${label} ready`);
          }
          return;
        }
        if (processingStatus === "error") {
          const detail = doc.metadata?.processing_error || "Document processing failed";
          throw new Error(detail);
        }
      }
      if (label) {
        setStatus(`${label} processing...`);
      }
      await new Promise(resolve => setTimeout(resolve, intervalMs));
    }

    ensureCurrent();
    if (lastDocs.length) {
      setDocuments(lastDocs);
    }
    throw new Error("Document processing timed out");
  };

  const handleDeleteDocument = async (id: string, title: string) => {
    const confirmed = window.confirm(`Delete "${title}"? This cannot be undone.`);
    if (!confirmed) {
      return;
    }
    // Captured after confirm: the delete goes to the connection applied when the
    // user confirmed it, and only that connection's UI may hear about it.
    const scope = currentScope();
    try {
      const response = await scopedFetch(scope, `/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      if (!isCurrentScope(scope)) return;
      setStatus(`Deleted ${title}`);
      toast({ title: "Document deleted", description: title, variant: "success" });
      void refreshAll(scope);
    } catch (error) {
      if (!isCurrentScope(scope)) return;
      setStatus(`Delete failed: ${toMessage(error)}`);
      toast({ title: "Delete failed", description: toMessage(error), variant: "error" });
    }
  };

  const handleDeleteAllDocuments = async () => {
    if (!window.confirm("Are you sure you want to clear ALL ingested documents? This cannot be undone.")) {
      return;
    }
    const scope = currentScope();
    setStatus("Clearing knowledge base...");
    try {
      const response = await scopedFetch(scope, "/documents", {
        method: "DELETE",
        headers: JSON_HEADERS,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      if (!isCurrentScope(scope)) return;
      setStatus("Knowledge base cleared");
      toast({ title: "Knowledge base cleared", description: "All documents deleted", variant: "success" });
      void refreshAll(scope);
    } catch (error) {
      if (!isCurrentScope(scope)) return;
      setStatus(`Clear failed: ${toMessage(error)}`);
      toast({ title: "Clear failed", description: toMessage(error), variant: "error" });
    }
  };

  // The scope is required: a refresh always reads through, and may only apply
  // to, the connection its operation started on.
  const refreshAll = async (scope: ConnectionScope, signal?: AbortSignal) => {
    const isCurrent = () => !signal?.aborted && isCurrentScope(scope);
    if (!isCurrent()) return;

    type Settled<T> = { ok: true; value: T } | { ok: false; error: unknown; aborted: boolean };

    const loadOne = async <T,>(path: string): Promise<Settled<T>> => {
      try {
        const value = await fetchJson<T>(scope, path, { signal });
        return { ok: true, value };
      } catch (error) {
        const aborted = Boolean(signal?.aborted) || (error instanceof DOMException && error.name === "AbortError");
        return { ok: false, error, aborted };
      }
    };

    // Each independently authorized load applies as soon as it settles. A hung
    // or 403 admin request must not block read-scoped documents.
    const failures: string[] = [];
    let applied = 0;
    let sawAbort = false;

    const apply = async <T,>(
      path: string,
      label: string,
      onSuccess: (value: T) => void,
    ) => {
      const result = await loadOne<T>(path);
      if (!isCurrent()) return;
      if (!result.ok) {
        if (result.aborted) {
          sawAbort = true;
          return;
        }
        failures.push(`${label} (${toMessage(result.error)})`);
        return;
      }
      onSuccess(result.value);
      applied += 1;
    };

    await Promise.all([
      apply<DocumentOut[]>("/documents", "documents", value => {
        setDocuments(value);
      }),
      apply<AppConfig>("/config", "config", value => {
        setConfig(value);
        setSelectedProfile(value.profile);
        setNewProfileName(value.profile);
      }),
      apply<TraceOut[]>("/monitoring/traces", "traces", value => {
        setTraces(value);
      }),
      apply<CacheStats>("/monitoring/cache", "cache", value => {
        setCacheStats(value);
      }),
      apply<SecurityPostureResponse>("/security/posture", "security posture", value => {
        setSecurityPosture(value);
      }),
    ]);

    if (!isCurrent() || sawAbort) return;

    if (failures.length === 0) {
      setStatus("API data loaded");
    } else if (applied > 0) {
      setStatus(`Partial API data loaded; unavailable: ${failures.join(", ")}`);
    } else {
      setStatus(`Failed to load data: ${failures.join(", ")}`);
    }
  };

  const refreshOnboarding = async (scope: ConnectionScope, signal?: AbortSignal) => {
    const owns = () => !signal?.aborted && isCurrentScope(scope);
    if (!owns()) return;
    try {
      const data = await fetchJson<OnboardingState>(scope, "/onboarding", { signal });
      if (!owns()) return;
      setOnboarding(data);
    } catch (error) {
      if (!owns()) return;
      setStatus(`Onboarding failed to load: ${toMessage(error)}`);
    }
  };

  const refreshModelStatus = async (scope: ConnectionScope, embeddingModel?: string, rerankerModel?: string) => {
    if (!embeddingModel && !rerankerModel) {
      return;
    }
    if (!isCurrentScope(scope)) return;
    const finishActivity = activities.begin(scope, setIsCheckingModels);
    try {
      const status = await fetchJson<ModelStatus>(scope, "/config/models/status", {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify({
          embedding_model: embeddingModel,
          reranker_model: rerankerModel,
        }),
      });
      if (!isCurrentScope(scope)) return;
      setModelStatus(status);
    } catch (error) {
      if (!isCurrentScope(scope)) return;
      setModelStatus({
        embedding: "error",
        reranker: "error",
        embedding_message: toMessage(error),
        reranker_message: toMessage(error),
      });
    } finally {
      finishActivity();
    }
  };

  const downloadInstallReport = async () => {
    const scope = currentScope();
    const finishActivity = activities.begin(scope, setIsExportingInstallReport);
    try {
      const report = await fetchJson<InstallReportResponse>(scope, "/install/report");
      // A report for a replaced connection is not offered under the new one.
      if (!isCurrentScope(scope)) return;
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      const timestamp = report.generated_at.replace(/[:.]/g, "-");
      link.href = url;
      link.download = `jr-autorag-install-report-${timestamp}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      toast({ title: "Install report exported", description: report.status, variant: "success" });
    } catch (error) {
      if (!isCurrentScope(scope)) return;
      setStatus(`Install report export failed: ${toMessage(error)}`);
      toast({ title: "Install report export failed", description: toMessage(error), variant: "error" });
    } finally {
      finishActivity();
    }
  };

  const downloadModel = async (kind: "embedding" | "reranker") => {
    if (!config?.retrieval) {
      return;
    }
    const model =
      kind === "embedding" ? config.retrieval.embedding_model : config.retrieval.reranker_model;
    if (!model) {
      return;
    }
    const scope = currentScope();
    const { embedding_model: embeddingModel, reranker_model: rerankerModel } = config.retrieval;
    const finishActivity = activities.begin(scope, kind === "embedding" ? setIsDownloadingEmbedding : setIsDownloadingReranker);
    try {
      await fetchJson(scope, "/config/models/download", {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify({ kind, model }),
      });
      if (!isCurrentScope(scope)) return;
      const message = `Downloaded ${kind} model`;
      setStatus(message);
      setModelActionMessage(message);
      await refreshModelStatus(scope, embeddingModel, rerankerModel);
    } catch (error) {
      if (!isCurrentScope(scope)) return;
      const message = `Download failed: ${toMessage(error)}`;
      setStatus(message);
      setModelActionMessage(message);
    } finally {
      finishActivity();
    }
  };

  const deleteModel = async (kind: "embedding" | "reranker") => {
    if (!config?.retrieval) {
      return;
    }
    const model =
      kind === "embedding" ? config.retrieval.embedding_model : config.retrieval.reranker_model;
    if (!model) {
      return;
    }
    const confirmed = window.confirm(`Remove cached ${kind} model "${model}"?`);
    if (!confirmed) {
      return;
    }
    const scope = currentScope();
    const { embedding_model: embeddingModel, reranker_model: rerankerModel } = config.retrieval;
    try {
      await fetchJson(scope, "/config/models/delete", {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify({ kind, model }),
      });
      if (!isCurrentScope(scope)) return;
      const message = `Removed ${kind} model cache`;
      setStatus(message);
      setModelActionMessage(message);
      await refreshModelStatus(scope, embeddingModel, rerankerModel);
    } catch (error) {
      if (!isCurrentScope(scope)) return;
      const message = `Remove failed: ${toMessage(error)}`;
      setStatus(message);
      setModelActionMessage(message);
    }
  };

  useEffect(() => {
    const controller = new AbortController();
    const scope = currentScope();
    void refreshAll(scope, controller.signal);
    void refreshOnboarding(scope, controller.signal);
    void refreshLocalProviders(scope, controller.signal);
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl, apiKey, connectAttempt]);

  useEffect(() => {
    setLocalSelections(prev => {
      const next = { ...prev };
      localProviders.forEach(provider => {
        if (!next[provider.base_url]) {
          const fallback = provider.running[0] ?? provider.models[0] ?? "";
          next[provider.base_url] = {
            planner: fallback,
            gatherer: fallback,
            generator: fallback,
          };
        }
      });
      return next;
    });
  }, [localProviders]);

  useEffect(() => {
    setSelectedDocumentIds(prev => prev.filter(id => documents.some(doc => doc.id === id)));
  }, [documents]);

  useEffect(() => {
    if (!config?.retrieval) {
      return;
    }
    void refreshModelStatus(
      currentScope(),
      config.retrieval.embedding_model,
      config.retrieval.reranker_model,
    );
  }, [config?.retrieval?.embedding_model, config?.retrieval?.reranker_model]);

  // Editing the draft while Connect is pending cancels that attempt, so a
  // superseded attempt never applies and never overwrites the newer draft.
  const cancelPendingConnect = () => {
    if (!pendingConnect.current) return;
    pendingConnect.current.abort();
    pendingConnect.current = null;
    setIsTestingConnection(false);
  };

  const handleDraftBaseUrlChange = (value: string) => {
    cancelPendingConnect();
    // A key is only valid for the origin it was entered for. When the draft
    // moves to another origin, drop the draft key so it is never sent there
    // implicitly; when it moves back to the applied origin, restore the applied key.
    const nextOrigin = originOf(value);
    if (nextOrigin !== originOf(draftBaseUrl)) {
      setDraftApiKey(nextOrigin === originOf(baseUrl) ? apiKey : "");
    }
    setDraftBaseUrl(value);
  };

  const handleTestConnection = async () => {
    const nextBaseUrl = normalizeApiBaseUrl(draftBaseUrl);
    const nextApiKey = draftApiKey.trim();
    // Retrying the applied URL and key doesn't restart the heartbeat, so the
    // retry takes a probe number and older ticks can't overwrite its result.
    const retriesApplied = nextBaseUrl === baseUrl && nextApiKey === apiKey;
    const probe = retriesApplied ? ++latestStatusProbe.current : 0;
    const ownsStatus = () => !retriesApplied || probe === latestStatusProbe.current;
    pendingConnect.current?.abort();
    const controller = new AbortController();
    pendingConnect.current = controller;
    setIsTestingConnection(true);
    setStatus(`Connecting to ${nextBaseUrl}...`);
    const result = await checkConnection(nextBaseUrl, nextApiKey, controller.signal);
    if (controller.signal.aborted) {
      return;
    }
    pendingConnect.current = null;
    setIsTestingConnection(false);
    if (!result.ok) {
      // The previous connection stays applied and the drafts stay editable.
      if (retriesApplied && ownsStatus()) {
        setIsConnected(false);
      }
      const message = retriesApplied
        ? `${nextBaseUrl} ${result.reason}.`
        : `${nextBaseUrl} ${result.reason}. Still using ${baseUrl}.`;
      setStatus(`Connect failed: ${message}`);
      toast({ title: "Connect failed", description: message, variant: "error" });
      return;
    }
    setStatus(`Connected to ${nextBaseUrl}`);
    if (ownsStatus()) {
      setIsConnected(true);
    }
    toast({ title: "Connected", description: nextBaseUrl, variant: "success" });
    // Data loaded with another URL or key must not be shown under this one,
    // for example admin-only config after switching to a read-only key.
    if (!retriesApplied) {
      appliedConnection.current = Object.freeze({
        generation: appliedConnection.current.generation + 1,
        baseUrl: nextBaseUrl,
        apiKey: nextApiKey,
      });
      activities.reset();
      discardActiveQuery();
      setConfig(null);
      setDocuments([]);
      setTraces([]);
      setCacheStats(null);
      setSecurityPosture(null);
      setOnboarding(null);
      setLocalProviders([]);
      setLocalSelections({});
      setSelectedDocumentIds([]);
    }
    setBaseUrl(nextBaseUrl);
    setApiKey(nextApiKey);
    setDraftBaseUrl(nextBaseUrl);
    setDraftApiKey(nextApiKey);
    setConnectAttempt(attempt => attempt + 1);
  };

  const handleSelectProfile = (name: string) => {
    setSelectedProfile(name);
    setConfig(cfg => {
      if (!cfg) {
        return cfg;
      }
      const profile = cfg.provider_profiles?.find(p => p.name === name);
      return profile ? { ...cfg, profile: name, provider: profile.provider } : cfg;
    });
  };

  const handleAddProfile = () => {
    if (!config?.provider || !newProfileName.trim()) {
      return;
    }
    const profile: ProviderProfile = { name: newProfileName.trim(), provider: config.provider };
    setConfig(cfg =>
      cfg
        ? {
          ...cfg,
          provider_profiles: [...(cfg.provider_profiles ?? []), profile],
          profile: profile.name,
        }
        : cfg,
    );
    setSelectedProfile(profile.name);
    setStatus(`Saved profile "${profile.name}"`);
    toast({ title: "Profile saved", description: profile.name, variant: "success" });
  };

  const persistConfigIn = async (scope: ConnectionScope, nextConfig: AppConfig, message = "Configuration saved") => {
    const finishActivity = activities.begin(scope, setIsSavingConfig);
    try {
      const updated = await fetchJson<AppConfig>(scope, "/config", {
        method: "PUT",
        headers: JSON_HEADERS,
        body: JSON.stringify({ ...nextConfig, profile: selectedProfile }),
      });
      if (!isCurrentScope(scope)) return;
      setConfig(updated);
      setStatus(message);
      toast({ title: "Configuration saved", description: nextConfig.provider?.name || "Provider updated", variant: "success" });
    } catch (error) {
      if (!isCurrentScope(scope)) throw error;
      console.error("Error saving configuration:", error);
      setStatus(`Save failed: ${toMessage(error)}`);
      toast({ title: "Save failed", description: toMessage(error), variant: "error" });
      throw error;
    } finally {
      finishActivity();
    }
  };
  const persistConfig = (nextConfig: AppConfig, message?: string) =>
    persistConfigIn(currentScope(), nextConfig, message);

  const handleSaveConfig = async () => {
    if (!config) {
      return;
    }
    await persistConfig(config);
  };

  const handleDiscoverModels = async () => {
    if (!config?.provider) {
      return;
    }
    const scope = currentScope();
    setStatus("Discovering models...");
    try {
      const models = await fetchJson<string[]>(scope, "/config/models", {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify(config.provider),
      });
      if (!isCurrentScope(scope)) return;
      setModelOptions(models);
      setStatus(`Found ${models.length} models`);
      toast({ title: "Models discovered", description: `${models.length} options found`, variant: "success" });
    } catch (error) {
      if (!isCurrentScope(scope)) return;
      console.error("Error discovering models:", error);
      setStatus(`Model discovery failed: ${toMessage(error)}`);
      toast({ title: "Model discovery failed", description: toMessage(error), variant: "error" });
    }
  };

  const handleIngest = async () => {
    const scope = currentScope();
    const finishActivity = activities.begin(scope, setIsIngesting);
    try {
      const payload: Record<string, unknown> = {
        title: ingestTitle,
        text: ingestText,
        sync: ingestSync,
        ocr_policy: config?.ingest?.ocr?.policy,
      };
      if (langextractProfileOverride !== "__global__") {
        payload.langextract_profile_override = langextractProfileOverride;
      }
      const promptOverride = langextractPromptOverride.trim();
      if (promptOverride) {
        payload.langextract_prompt_override = promptOverride;
      }
      const result = await fetchJson<IngestResponse>(scope, "/documents/text", {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify(payload),
      });
      if (!isCurrentScope(scope)) return;
      setStatus(`Processing ${result.title}...`);
      setIngestText("");
      await waitForDocumentReady(scope, result.document_id, result.title);
      if (!isCurrentScope(scope)) return;
      toast({ title: "Ingest queued", description: result.title, variant: "success" });
    } catch (error) {
      if (!isCurrentScope(scope)) return;
      setStatus(`Ingest failed: ${toMessage(error)}`);
      toast({ title: "Ingest failed", description: toMessage(error), variant: "error" });
    } finally {
      finishActivity();
    }
  };

  const handleAsk = async (overrideQuestion?: string, isRegenerate = false) => {
    if (isQuerying) return; // Prevent double trigger

    // 1. Determine the question to ask
    const currentQuestion = overrideQuestion || question;
    if (!currentQuestion.trim()) return;
    const queryId = ++queryIdSeq.current;
    const queryScope = currentScope();
    const controller = new AbortController();
    const queryTransport: ActiveQueryTransport = {
      queryId,
      scope: queryScope,
      controller,
    };
    activeQueryTransport.current = queryTransport;
    const ownsQuery = () => activeQueryTransport.current?.queryId === queryId;
    const connectionStillCurrent = () => isCurrentScope(queryScope);
    const mayApplyQueryUi = () => ownsQuery() && connectionStillCurrent();

    // NEW: Session persistence logic
    let localSessionId = currentSessionId;
    let initialHistory = chatHistory;

    if (isRegenerate) {
      // Remove last assistant message if it exists
      initialHistory = chatHistory.slice(0, chatHistory[chatHistory.length - 1]?.role === "assistant" ? -1 : undefined);
      setChatHistory(initialHistory);
    } else {
      // Persist new user question immediately
      initialHistory = [...chatHistory, { role: "user", content: currentQuestion }];
      setChatHistory(initialHistory);
      setQuestion(""); // Clear input immediately
    }

    if (!localSessionId) {
      localSessionId = crypto.randomUUID();
      setCurrentSessionId(localSessionId);
      const firstMsg = currentQuestion;
      const title = firstMsg.length > 40 ? firstMsg.slice(0, 37) + "..." : firstMsg;
      const newSession: ChatSession = {
        id: localSessionId,
        title,
        history: initialHistory,
        queryResult: null,
        createdAt: new Date().toISOString(),
      };
      setSavedSessions(prev => [newSession, ...prev]);
    } else {
      // Update existing session in sidebar immediately with user message
      setSavedSessions(prev => prev.map(s => s.id === localSessionId ? { ...s, history: initialHistory } : s));
    }

    setIsQuerying(true);
    setQueryResult(null);
    setActiveStage("planning");
    setProgress(null);

    try {
      // Use currentQuestion instead of state 'question'
      const useFilter = selectedDocumentIds.length > 0 && selectedDocumentIds.length < documents.length;

      // Filter chat history to exclude the last assistant message if we are regenerating
      let historyToSend = initialHistory;
      const lastItem = historyToSend.length > 0 ? historyToSend[historyToSend.length - 1] : null;
      if (isRegenerate && lastItem && lastItem.role === "assistant") {
        historyToSend = historyToSend.slice(0, -1);
      }

      const payload = useFilter
        ? { question: currentQuestion, document_ids: selectedDocumentIds, history: historyToSend, conversation_id: localSessionId, query_mode: queryMode }
        : { question: currentQuestion, history: historyToSend, conversation_id: localSessionId, query_mode: queryMode };
      const response = await scopedFetch(queryScope, "/query/stream", {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify(payload),
        signal: controller.signal,
      });
      if (!response.ok || !response.body) {
        throw new Error(await response.text());
      }

      // Connection status stays with heartbeat/Connect ownership. A late stream
      // after a failed probe or a connection switch must not restore Connected.
      if (!mayApplyQueryUi()) {
        return;
      }

      const body = response.body;
      if (!body) {
        throw new Error("Response body is null");
      }

      const reader = body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let receivedResult = false;

      while (true) {
        if (controller.signal.aborted || !ownsQuery()) {
          try {
            await reader.cancel();
          } catch {
            // Reader may already be closed by fetch abort.
          }
          break;
        }
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let boundary = buffer.indexOf("\n\n");
        while (boundary !== -1) {
          const chunk = buffer.slice(0, boundary).trim();
          buffer = buffer.slice(boundary + 2);
          boundary = buffer.indexOf("\n\n");
          if (!chunk.startsWith("data:")) {
            continue;
          }
          const payloadText = chunk.replace(/^data:\s*/, "");
          let event: { type?: string; data?: any };
          try {
            event = JSON.parse(payloadText);
          } catch {
            continue;
          }
          if (event.type === "step") {
            if (mayApplyQueryUi() && localSessionId === currentSessionIdRef.current) {
              setQueryResult(prev => {
                const base: QueryResponse = prev ?? {
                  answer: "",
                  chunks: [],
                  sources: [],
                  trace_id: "",
                  metrics: {},
                  steps: [],
                };
                return {
                  ...base,
                  steps: [...(base.steps ?? []), event.data],
                };
              });
            }
          }
          if (event.type === "token") {
            const text = event.data?.text ?? "";
            if (text) {
              if (mayApplyQueryUi() && localSessionId === currentSessionIdRef.current) {
                setQueryResult(prev => {
                  const base: QueryResponse = prev ?? {
                    answer: "",
                    chunks: [],
                    sources: [],
                    trace_id: "",
                    metrics: {},
                    steps: [],
                  };
                  return {
                    ...base,
                    answer: `${base.answer}${text}`,
                  };
                });
              }
            }
          }
          if (event.type === "stage") {
            if (mayApplyQueryUi() && localSessionId === currentSessionIdRef.current) {
              setActiveStage(event.data?.name ?? null);
            }
          }
          if (event.type === "progress") {
            if (mayApplyQueryUi() && localSessionId === currentSessionIdRef.current) {
              setProgress(event.data);
              if (event.data?.trace_id) {
                setCurrentTraceId(event.data.trace_id);
              }
            }
          }

          if (event.type === "result") {
            receivedResult = true;
            const updatedHistory = [
              ...initialHistory,
              { role: "assistant", content: event.data.answer }
            ];

            if (mayApplyQueryUi() && localSessionId === currentSessionIdRef.current) {
              setQueryResult(event.data);
              setStatus("Query succeeded");
              setActiveStage(null);
              setProgress(null);
              setChatHistory(updatedHistory);
            }

            // Session list keeps the answer under its session id; do not refresh
            // API data or toast when this stream's connection was replaced.
            if (mayApplyQueryUi()) {
              setSavedSessions(prev => prev.map(s =>
                s.id === localSessionId ? { ...s, history: updatedHistory, queryResult: event.data } : s
              ));
              void refreshAll(queryScope);
              toast({ title: "Query complete", description: "Results updated", variant: "success" });
            }
          }
          if (event.type === "error") {
            const message = event.data?.message ?? "Query failed";
            throw new Error(message);
          }
        }
      }

      if (controller.signal.aborted || !ownsQuery()) {
        return;
      }

      if (!receivedResult) {
        throw new Error("Query ended before returning a result");
      }
    } catch (error) {
      if (controller.signal.aborted || !ownsQuery()) {
        return;
      }
      console.error("Query execution error:", error);
      setStatus(`Query failed: ${toMessage(error)}`);

      // Attempt to save partial result if available
      setQueryResult((currentPartial) => {
        if (mayApplyQueryUi() && currentPartial?.answer && localSessionId === currentSessionIdRef.current) {
          const partialAnswer = currentPartial.answer + "\n\n*[Generation interrupted due to error]*";
          const updatedHistory = [
            ...initialHistory,
            { role: "assistant", content: partialAnswer }
          ];
          setChatHistory(updatedHistory);
          setSavedSessions(prev => prev.map(s =>
            s.id === localSessionId ? { ...s, history: updatedHistory } : s
          ));
        }
        return currentPartial;
      });

      toast({ title: "Query interrupted", description: toMessage(error), variant: "error" });
    } finally {
      if (activeQueryTransport.current?.queryId === queryId) {
        activeQueryTransport.current = null;
        setIsQuerying(false);
        setActiveStage(null);
        setProgress(null);
        setCurrentTraceId(null);
      }
    }
  };

  const handleCancelQuery = async () => {
    const transport = activeQueryTransport.current;
    if (!transport) return;
    const cancelledQueryId = transport.queryId;
    const traceId = currentTraceId;
    const cancelPath = `/query/cancel${traceId ? `?trace_id=${encodeURIComponent(traceId)}` : ""}`;

    // Stop the local stream immediately and release ownership so a same-connection
    // Ask can start. A delayed cancel response must not clear that newer query.
    if (activeQueryTransport.current?.queryId === cancelledQueryId) {
      discardActiveQuery();
    } else {
      transport.controller.abort();
    }

    if (!traceId) {
      // No trace id yet, so there is nothing to ask the server to cancel. Only the
      // local stream stopped; the server may still finish this query.
      const message = "Stopped streaming. The server had not reported this query yet, so it was not asked to cancel.";
      setStatus(message);
      toast({ title: "Stopped streaming", description: message, variant: "info" });
      return;
    }

    // A late cancel reply may only report on the connection and query it was sent
    // for: not after a newer Ask, and not after Connect applied another URL/key.
    const noNewerQuery = () => queryIdSeq.current === cancelledQueryId && isCurrentScope(transport.scope);

    try {
      const response = await scopedFetch(transport.scope, cancelPath, { method: "POST" });
      if (!response.ok) {
        const detail = await response.text().catch(() => "");
        const message = detail.trim() || `Cancel failed (${response.status})`;
        if (noNewerQuery()) {
          setStatus(`Cancel failed: ${message}`);
          toast({ title: "Cancel failed", description: message, variant: "error" });
        }
        return;
      }
      if (noNewerQuery()) {
        setStatus("Query cancelled");
        toast({ title: "Query cancelled", variant: "info" });
      }
    } catch (e) {
      console.error("Cancel failed", e);
      if (noNewerQuery()) {
        setStatus(`Cancel failed: ${toMessage(e)}`);
        toast({ title: "Cancel failed", description: toMessage(e), variant: "error" });
      }
    }
  };

  const handleNewChat = () => {
    setChatHistory([]);
    setQueryResult(null);
    setCurrentSessionId(null);
    setActiveStage(null);
    setProgress(null);
    setQuestion("");
    toast({ title: "New chat started", variant: "info" });
  };

  const handleClearHistory = () => {
    if (window.confirm("Delete all saved chat history? This cannot be undone.")) {
      setSavedSessions([]);
      toast({ title: "History cleared", variant: "success" });
    }
  };

  const handleLoadSession = (session: ChatSession) => {
    setCurrentSessionId(session.id);
    setChatHistory(session.history);
    setQueryResult(session.queryResult);
    setActiveTab("query");
    toast({ title: "Session restored", description: session.title });
  };

  const handleDeleteSession = (id: string) => {
    setSavedSessions(prev => prev.filter(s => s.id !== id));
  };

  const handleEvaluation = async () => {
    const scope = currentScope();
    const finishActivity = activities.begin(scope, setIsEvaluating);
    try {
      const payload = {
        name: "Smoke Test",
        questions: ["What is JR AutoRAG?", "How do I onboard documents?"],
      };
      const result = await fetchJson<{ average_coverage: number; average_tokens: number }>(scope, "/evaluation", {
        method: "POST",
        headers: JSON_HEADERS,
        body: JSON.stringify(payload),
      });
      if (!isCurrentScope(scope)) return;
      setEvaluationSummary(
        `Avg coverage ${(result.average_coverage * 100).toFixed(1)}%, Avg tokens ${result.average_tokens.toFixed(0)}`,
      );
      setStatus("Evaluation complete");
      toast({ title: "Evaluation complete", description: "Coverage and tokens updated", variant: "success" });
    } catch (error) {
      if (!isCurrentScope(scope)) return;
      const message = `Evaluation failed: ${toMessage(error)}`;
      setEvaluationSummary(message);
      setStatus(message);
      toast({ title: "Evaluation failed", description: toMessage(error), variant: "error" });
    } finally {
      finishActivity();
    }
  };

  const handleSeedDemo = async () => {
    const scope = currentScope();
    const finishActivity = activities.begin(scope, setIsSeedingDemo);
    setStatus("Loading demo corpus...");
    try {
      const result = await fetchJson<DemoSeedResponse>(scope, "/onboarding/demo/seed", {
        method: "POST",
        headers: JSON_HEADERS,
      });
      if (!isCurrentScope(scope)) return;
      setStatus(`Demo corpus ready with ${result.document_count} document(s)`);
      toast({
        title: "Demo corpus ready",
        description: `${result.seeded.length} added, ${result.skipped.length} already present`,
        variant: "success",
      });
      await refreshAll(scope);
      await refreshOnboarding(scope);
    } catch (error) {
      if (!isCurrentScope(scope)) return;
      setStatus(`Demo setup failed: ${toMessage(error)}`);
      toast({ title: "Demo setup failed", description: toMessage(error), variant: "error" });
    } finally {
      finishActivity();
    }
  };

  const handleAskExample = (query: string) => {
    setActiveTab("query");
    setQuestion(query);
    if (documents.length > 0 && !isQuerying) {
      void handleAsk(query);
    }
  };

  const handleClearCache = async () => {
    const confirmed = window.confirm(
      "Clear all cached indexes and query results? This will require a rebuild on next query."
    );
    if (!confirmed) {
      return;
    }
    const scope = currentScope();
    const finishActivity = activities.begin(scope, setIsClearingCache);
    try {
      const result = await fetchJson<{ status: string; message: string }>(scope, "/api/cache/clear", {
        method: "DELETE",
      });
      if (!isCurrentScope(scope)) return;
      setStatus(result.message || "Cache cleared");
      void refreshAll(scope);
      toast({ title: "Cache cleared", description: "All indexes and query caches reset", variant: "success" });
    } catch (error) {
      if (!isCurrentScope(scope)) return;
      setStatus(`Cache clear failed: ${toMessage(error)}`);
      toast({ title: "Cache clear failed", description: toMessage(error), variant: "error" });
    } finally {
      finishActivity();
    }
  };

  const setLocalSelection = (baseUrl: string, field: keyof RoleSelection, value: string) => {
    setLocalSelections(prev => ({
      ...prev,
      [baseUrl]: {
        planner: prev[baseUrl]?.planner ?? "",
        gatherer: prev[baseUrl]?.gatherer ?? "",
        generator: prev[baseUrl]?.generator ?? "",
        [field]: value,
      },
    }));
  };

  const applyLocalProvider = async (provider: LocalProviderInfo) => {
    const scope = currentScope();
    const selection = (localSelections[provider.base_url] || {}) as Partial<RoleSelection>;
    const ensureConfig = async () => {
      if (config) {
        return config;
      }
      setStatus("Loading configuration...");
      try {
        const cfg = await fetchJson<AppConfig>(scope, "/config");
        if (!isCurrentScope(scope)) return null;
        setConfig(cfg);
        setSelectedProfile(cfg.profile);
        setNewProfileName(cfg.profile);
        return cfg;
      } catch (error) {
        if (!isCurrentScope(scope)) return null;
        setStatus(`Failed to load configuration: ${toMessage(error)}`);
        toast({ title: "Load config failed", description: toMessage(error), variant: "error" });
        return null;
      }
    };

    const activeConfig = await ensureConfig();
    if (!activeConfig) {
      return;
    }
    if (!provider.models.length) {
      setStatus(`No models available for ${provider.name}. Install or run one first.`);
      toast({ title: "No models available", description: provider.name, variant: "error" });
      return;
    }

    // Smart fallback: try to find a running model if no selection made
    const firstRunning = provider.running.length > 0 ? provider.running[0] : provider.models[0];
    const planner = selection.planner || firstRunning;
    const generator = selection.generator || firstRunning;
    const gatherer = selection.gatherer || firstRunning;

    const nextConfig: AppConfig = {
      ...activeConfig,
      provider: {
        name: provider.name,
        base_url: provider.base_url,
        planner_model: planner,
        generator_model: generator,
        gatherer_model: gatherer,
      },
    };
    if (!isCurrentScope(scope)) return;
    setConfig(nextConfig);
    try {
      await persistConfigIn(scope, nextConfig, `Applied ${provider.name} (${planner}/${generator})`);
      if (!isCurrentScope(scope)) return;
      setStatus(`Successfully applied ${provider.name} with ${planner}`);
      toast({ title: "Provider applied", description: `${provider.name} (${planner}/${generator})`, variant: "success" });
    } catch {
      // status already updated inside persistConfig
    }
  };

  // Sends one file on the batch's scope. Refuses to send once that connection
  // has been replaced, so a stale batch can never reach the new destination.
  const uploadFileIn = async (scope: ConnectionScope, file: File): Promise<IngestResponse> => {
    if (!isCurrentScope(scope)) throw new StaleConnectionError();
    const finishActivity = activities.begin(scope, setIsUploadingFile);
    setStatus(`Uploading ${file.name}...`);
    try {
      const formData = new FormData();
      const title = file.name.replace(/\.[^.]+$/, "");
      formData.append("title", title || file.name);
      formData.append("file", file);
      formData.append("sync", String(ingestSync));
      if (config?.ingest?.ocr?.policy) {
        formData.append("ocr_policy", config.ingest.ocr.policy);
      }
      if (langextractProfileOverride !== "__global__") {
        formData.append("langextract_profile_override", langextractProfileOverride);
      }
      const promptOverride = langextractPromptOverride.trim();
      if (promptOverride) {
        formData.append("langextract_prompt_override", promptOverride);
      }
      const resp = await scopedFetch(scope, "/documents/upload", {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || "Upload failed");
      }
      const result: IngestResponse = await resp.json();
      if (!isCurrentScope(scope)) throw new StaleConnectionError();
      setStatus(`Processing ${file.name}...`);
      setIngestTitle("");
      setIngestText("");
      return result;
    } catch (error) {
      if (!isCurrentScope(scope)) throw error;
      setStatus(`Upload failed: ${toMessage(error)}`);
      toast({ title: "Upload failed", description: toMessage(error), variant: "error" });
      throw error;
    } finally {
      finishActivity();
      if (isCurrentScope(scope) && fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const startUploadBatch = (): UploadBatch => {
    const scope = currentScope();
    return Object.freeze({
      isCurrent: () => isCurrentScope(scope),
      upload: (file: File) => uploadFileIn(scope, file),
      waitUntilReady: (upload: IngestResponse, label?: string) =>
        waitForDocumentReady(scope, upload.document_id, label),
    });
  };

  const updateRetrieval = (field: keyof RetrievalDefaults, value: string | number | boolean) => {
    setConfig(cfg => (cfg ? { ...cfg, retrieval: { ...cfg.retrieval, [field]: value } } : cfg));
  };

  const updateDeploymentProfile = (value: AppConfig["deployment_profile"]) => {
    setConfig(cfg => (cfg ? { ...cfg, deployment_profile: value } : cfg));
  };

  const updateIngestOcrPolicy = (value: OCRPolicy) => {
    setConfig(cfg =>
      cfg
        ? {
            ...cfg,
            ingest: {
              ...cfg.ingest,
              ocr: {
                ...cfg.ingest.ocr,
                policy: value,
              },
            },
          }
        : cfg,
    );
  };

  const updateBackend = (subsystem: string, patch: Partial<SubsystemBackendConfig>) => {
    setConfig(cfg =>
      cfg
        ? (() => {
            const current = cfg.backends[subsystem];
            if (!current) {
              return cfg;
            }
            return {
              ...cfg,
              backends: {
                ...cfg.backends,
                [subsystem]: {
                  ...current,
                  ...patch,
                  subsystem: patch.subsystem ?? current.subsystem,
                  label: patch.label ?? current.label,
                  backend_id: patch.backend_id ?? current.backend_id,
                  enabled: patch.enabled ?? current.enabled,
                  settings: patch.settings ?? current.settings,
                  capabilities: {
                    ...current.capabilities,
                    ...patch.capabilities,
                  },
                },
              },
            };
          })()
        : cfg,
    );
  };

  const docsReady = documents.length > 0;
  const modelsReady = Boolean(config?.provider?.planner_model && config?.provider?.generator_model);
  const apiReady = isConnected;
  const securityLevel = securityPosture?.level ?? "needs_attention";
  const securityReady = securityLevel === "client_ready" || securityLevel === "local_only";
  const securityLabel = securityLevel.replace("_", " ");
  const securityBlockers = securityPosture?.checks.filter(check => check.status === "fail").length ?? 0;
  const readyChecks = [
    { label: "API", ready: apiReady, detail: apiReady ? "FastAPI reachable" : "Connect local API" },
    {
      label: "Models",
      ready: modelsReady,
      detail: modelsReady ? "Planner and generator selected" : "Choose local or routed models",
    },
    {
      label: "Corpus",
      ready: docsReady,
      detail: docsReady ? `${documents.length} document${documents.length === 1 ? "" : "s"} indexed` : "Ingest client documents",
    },
    {
      label: "Security",
      ready: securityReady,
      detail: securityPosture?.summary ?? "Load security posture",
    },
  ];
  const readyPercent = Math.round((readyChecks.filter(check => check.ready).length / readyChecks.length) * 100);
  const providerName = config?.provider?.name || "No provider selected";
  const retrievalMode = activePreset.replace("_", " ");
  const demoMode = onboarding?.demo_mode ?? false;
  const demoSeeded = Boolean(onboarding?.demo_seeded || onboarding?.demo_document_count || docsReady);

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-background transition-colors duration-300">
      <header className="flex-none relative z-50 border-b border-border/60 bg-background/95">
        <div className="mx-auto grid max-w-[1600px] grid-cols-[minmax(0,148px)_minmax(0,1fr)_auto] items-center gap-2 px-3 py-2 sm:grid-cols-[minmax(0,176px)_minmax(0,1fr)_auto] sm:gap-3 sm:px-4">
          <div className="flex min-w-0 items-center gap-2">
            <div className="flex min-w-0 items-center gap-2.5">
              <img
                src="/HWC-Icon.png"
                alt="JR AutoRAG"
                className="size-9 shrink-0 rounded-lg object-cover shadow-sm"
              />
              <div className="min-w-0">
                <h1 className="truncate text-lg font-bold leading-tight text-foreground sm:text-xl">JR AutoRAG</h1>
                <p className="hidden truncate text-xs text-muted-foreground sm:block">Local RAG Command Center</p>
              </div>
            </div>
          </div>

          {/* Top Navigation — ARIA tablist with roving tabindex + arrow keys */}
          <nav
            className="no-scrollbar flex min-w-0 items-center gap-1 overflow-x-auto rounded-lg bg-muted/30 p-1"
            role="tablist"
            aria-label="Main navigation"
            onKeyDown={(e) => {
              if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
                e.preventDefault();
                const currentIndex = tabs.findIndex(t => t.id === activeTab);
                const direction = e.key === "ArrowRight" ? 1 : -1;
                const nextIndex = (currentIndex + direction + tabs.length) % tabs.length;
                const nextTab = tabs[nextIndex];
                if (nextTab) {
                  setActiveTab(nextTab.id);
                  // Focus the newly selected tab button
                  requestAnimationFrame(() => {
                    const btn = document.querySelector<HTMLButtonElement>(`[data-tab-id="${nextTab.id}"]`);
                    btn?.focus();
                  });
                }
              }
            }}
          >
            {tabs.map(tab => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  data-tab-id={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  role="tab"
                  aria-selected={isActive}
                  aria-controls={`tabpanel-${tab.id}`}
                  aria-label={tab.label}
                  title={tab.label}
                  tabIndex={isActive ? 0 : -1}
                  className={`flex shrink-0 items-center gap-2 rounded-md px-2.5 py-1.5 text-sm font-medium transition-all ${isActive
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "text-muted-foreground hover:bg-muted hover:text-foreground"
                    }`}
                >
                  <Icon className="size-4 shrink-0" />
                  <span className="hidden sm:inline">{tab.label}</span>
                </button>
              );
            })}
          </nav>

          <div className="flex min-w-0 items-center justify-end gap-2">
            {/* API URL — visible on all screens, compact on smaller */}
            <form
              className="hidden lg:flex items-center gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                void handleTestConnection();
              }}
            >
              <Input
                className="w-40 2xl:w-48 text-xs"
                type="url"
                required
                value={draftBaseUrl}
                onChange={e => handleDraftBaseUrlChange(e.target.value)}
                placeholder="API URL"
                aria-label="API base URL"
              />
              <Input
                className="hidden 2xl:block w-40 text-xs"
                value={draftApiKey}
                onChange={e => {
                  cancelPendingConnect();
                  setDraftApiKey(e.target.value);
                }}
                type="password"
                autoComplete="off"
                placeholder="X-API-Key (session)"
                aria-label="API key"
              />
              <Button size="sm" variant="outline" type="submit" disabled={isTestingConnection}>
                {isTestingConnection ? "Connecting" : apiReady ? "Connected" : "Connect"}
              </Button>
            </form>

            {/* Checklist items indicator */}
            <div
              className="hidden items-center gap-1.5 rounded-full bg-muted/30 px-2.5 py-1.5 text-xs text-muted-foreground/80 xl:flex"
              aria-label={`Readiness: API ${apiReady ? "ready" : "needs attention"}, models ${modelsReady ? "ready" : "needs attention"}, knowledge ${docsReady ? "ready" : "needs attention"}, security ${securityReady ? "ready" : "needs attention"}`}
            >
              {readyChecks.map(check => (
                <div key={check.label} className="flex items-center gap-1.5">
                  <div className={`h-2 w-2 rounded-full ${check.ready ? "bg-primary" : "bg-secondary"}`} />
                  <span className="font-medium">{check.label === "Corpus" ? "Knowledge" : check.label}</span>
                </div>
              ))}
            </div>

            {/* Theme Toggle */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsDarkMode(!isDarkMode)}
              className="text-foreground"
              aria-label={isDarkMode ? "Switch to light mode" : "Switch to dark mode"}
            >
              {isDarkMode ? <Moon className="h-5 w-5" /> : <Sun className="h-5 w-5" />}
            </Button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 min-h-0 mx-auto w-full max-w-[1600px]">
        <div className="flex h-full flex-col">
          <div className={`flex-1 bg-background ${activeTab === "query" ? "overflow-hidden p-0" : "overflow-auto p-6"}`}>
            <div id="tabpanel-home" role="tabpanel" aria-labelledby="tab-home" tabIndex={0} className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "home" ? "hidden" : ""}`}>
              <div className="mx-auto flex max-w-[1500px] flex-col gap-6">
                <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_380px]">
                  <div className="rounded-lg border border-border/60 bg-card p-6 shadow-sm">
                    <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                      <div className="max-w-3xl">
                        <div className="mb-3 flex items-center gap-2 text-sm font-medium text-primary">
                          <ShieldCheck className="size-4" />
                          Local-first enterprise install
                        </div>
                        <h2 className="text-2xl font-semibold tracking-tight text-foreground sm:text-3xl">
                          Install private AutoRAG for real client knowledge bases.
                        </h2>
                        <p className="mt-3 max-w-2xl text-sm leading-6 text-muted-foreground">
                          Connect the local API, choose private model routes, ingest client documents, inspect retrieval evidence, and export the receipts needed for a B2B handoff.
                        </p>
                      </div>

                      <div className="grid min-w-[260px] gap-2 rounded-lg border border-border/60 bg-muted/10 p-4">
                        <div className="flex items-center justify-between gap-3">
                          <span className="text-sm font-medium text-muted-foreground">Install readiness</span>
                          <span className="font-mono text-2xl font-semibold text-foreground">{readyPercent}%</span>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full bg-muted">
                          <div className="h-full rounded-full bg-primary transition-all duration-300" style={{ width: `${readyPercent}%` }} />
                        </div>
                        <div className="text-xs leading-relaxed text-muted-foreground">
                          {demoMode ? "Demo mode is active for this run. Use persistent mode for client installs." : "Persistent local mode is active for client installs."}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-lg border border-border/60 bg-card p-5 shadow-sm">
                    <div className="mb-4 flex items-center justify-between gap-3">
                      <div>
                        <div className="text-sm font-semibold text-foreground">Install Path</div>
                        <div className="text-xs text-muted-foreground">Bring a client workspace online.</div>
                      </div>
                      <PackageCheck className="size-5 text-primary" />
                    </div>
                    <div className="grid gap-2">
                      <Button onClick={() => setActiveTab("documents")}>
                        <FileText className="size-4" data-icon="inline-start" />
                        Ingest Client Documents
                      </Button>
                      <div className="grid grid-cols-2 gap-2">
                        <Button variant="outline" onClick={() => setActiveTab("config")}>
                          <Settings className="size-4" data-icon="inline-start" />
                          Configure
                        </Button>
                        <Button variant="outline" onClick={() => setActiveTab("query")}>
                          <MessageSquare className="size-4" data-icon="inline-start" />
                          Query
                        </Button>
                      </div>
                      <Button variant="secondary" onClick={() => setActiveTab("quality")}>
                        <FileCheck2 className="size-4" data-icon="inline-start" />
                        Prove Readiness
                      </Button>
                    </div>
                  </div>
                </section>

                <section className="grid gap-4 lg:grid-cols-4">
                  {readyChecks.map(check => (
                    <div key={check.label} className="rounded-lg border border-border/60 bg-card p-5 shadow-sm">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="text-xs font-semibold uppercase text-muted-foreground">{check.label}</div>
                          <div className="mt-2 text-lg font-semibold text-foreground">{check.ready ? "Ready" : "Needs attention"}</div>
                          <div className="mt-1 text-sm leading-6 text-muted-foreground">{check.detail}</div>
                        </div>
                        <div className={`rounded-full border p-2 ${check.ready ? "border-primary/30 bg-primary/10 text-primary" : "border-border bg-muted/20 text-muted-foreground"}`}>
                          {check.label === "API" ? <Activity className="size-4" /> : check.label === "Corpus" ? <FileText className="size-4" /> : <ShieldCheck className="size-4" />}
                        </div>
                      </div>
                    </div>
                  ))}
                </section>

                <section className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
                  <Suspense fallback={<LoadingSpinner message="Loading onboarding..." />}>
                    <OnboardingFlow
                      onboarding={onboarding}
                      apiReady={apiReady}
                      docsReady={docsReady}
                      modelsReady={modelsReady}
                      isSeedingDemo={isSeedingDemo}
                      activeMode={onboardingMode}
                      onModeChange={setOnboardingMode}
                      onSeedDemo={handleSeedDemo}
                      onOpenTab={setActiveTab}
                      onAskExample={handleAskExample}
                    />
                  </Suspense>

                  <div className="grid content-start gap-4">
                    <div className="rounded-lg border border-border/60 bg-card p-5 shadow-sm">
                      <div className="mb-4 flex items-center justify-between gap-3">
                        <div>
                          <div className="text-sm font-semibold text-foreground">Delivery Evidence</div>
                          <div className="text-xs text-muted-foreground">Artifacts an installer can hand to a client.</div>
                        </div>
                        <FileCheck2 className="size-5 text-primary" />
                      </div>
                      <div className="grid gap-3 text-sm">
                        <div className="flex items-center justify-between gap-3 rounded-md border border-border/50 bg-muted/10 px-3 py-2">
                          <span className="text-muted-foreground">Install doctor</span>
                          <code className="rounded bg-muted px-2 py-1 text-xs text-foreground">bun run doctor</code>
                        </div>
                        <div className="flex items-center justify-between gap-3 rounded-md border border-border/50 bg-muted/10 px-3 py-2">
                          <span className="text-muted-foreground">Client handoff</span>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => void downloadInstallReport()}
                            disabled={!apiReady || isExportingInstallReport}
                          >
                            {isExportingInstallReport ? "Exporting" : "Export"}
                            <ArrowRight className="size-4" data-icon="inline-end" />
                          </Button>
                        </div>
                        <div className="flex items-center justify-between gap-3 rounded-md border border-border/50 bg-muted/10 px-3 py-2">
                          <span className="text-muted-foreground">Quality receipts</span>
                          <Button variant="ghost" size="sm" onClick={() => setActiveTab("quality")}>
                            Open
                            <ArrowRight className="size-4" data-icon="inline-end" />
                          </Button>
                        </div>
                        <div className="flex items-center justify-between gap-3 rounded-md border border-border/50 bg-muted/10 px-3 py-2">
                          <span className="text-muted-foreground">Runtime traces</span>
                          <Button variant="ghost" size="sm" onClick={() => setActiveTab("metrics")}>
                            Inspect
                            <ArrowRight className="size-4" data-icon="inline-end" />
                          </Button>
                        </div>
                      </div>
                    </div>

                    <div className="rounded-lg border border-border/60 bg-card p-5 shadow-sm">
                      <div className="mb-4 flex items-center justify-between gap-3">
                        <div>
                          <div className="text-sm font-semibold text-foreground">Security Posture</div>
                          <div className="text-xs text-muted-foreground">Pre-exposure checks for client installs.</div>
                        </div>
                        <ShieldCheck className={`size-5 ${securityReady ? "text-primary" : "text-destructive"}`} />
                      </div>
                      <div className="grid gap-3 text-sm">
                        <div className="flex items-center justify-between gap-3 rounded-md border border-border/50 bg-muted/10 px-3 py-2">
                          <span className="text-muted-foreground">Posture</span>
                          <span className={`rounded-full px-2 py-1 text-xs font-semibold capitalize ${securityReady ? "bg-primary/10 text-primary" : "bg-destructive/10 text-destructive"}`}>
                            {securityLabel}
                          </span>
                        </div>
                        <div className="flex items-center justify-between gap-3 rounded-md border border-border/50 bg-muted/10 px-3 py-2">
                          <span className="text-muted-foreground">Blocking checks</span>
                          <span className="font-mono text-foreground">{securityBlockers}</span>
                        </div>
                        <Button variant="outline" size="sm" onClick={() => setActiveTab("config")}>
                          Open Configuration
                          <ArrowRight className="size-4" data-icon="inline-end" />
                        </Button>
                      </div>
                    </div>

                    <div className="rounded-lg border border-border/60 bg-card p-5 shadow-sm">
                      <div className="mb-4 flex items-center gap-2">
                        <TerminalSquare className="size-5 text-primary" />
                        <div>
                          <div className="text-sm font-semibold text-foreground">Current Runtime</div>
                          <div className="text-xs text-muted-foreground">Local operator context.</div>
                        </div>
                      </div>
                      <dl className="grid gap-3 text-sm">
                        <div className="flex items-center justify-between gap-3">
                          <dt className="text-muted-foreground">Provider</dt>
                          <dd className="max-w-[220px] truncate font-medium text-foreground">{providerName}</dd>
                        </div>
                        <div className="flex items-center justify-between gap-3">
                          <dt className="text-muted-foreground">Preset</dt>
                          <dd className="font-medium capitalize text-foreground">{retrievalMode}</dd>
                        </div>
                        <div className="flex items-center justify-between gap-3">
                          <dt className="text-muted-foreground">Documents</dt>
                          <dd className="font-mono text-foreground">{documents.length}</dd>
                        </div>
                        <div className="flex items-center justify-between gap-3">
                          <dt className="text-muted-foreground">Traces</dt>
                          <dd className="font-mono text-foreground">{traces.length}</dd>
                        </div>
                      </dl>
                    </div>
                  </div>
                </section>
              </div>
            </div>

            <div id="tabpanel-config" role="tabpanel" aria-labelledby="tab-config" tabIndex={0} className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "config" ? "hidden" : ""}`}>
              {/* Configuration Panel */}
              <div className="space-y-6 max-w-[1600px] mx-auto">
                <div>
                  <h2 className="text-lg font-semibold tracking-tight">System Configuration</h2>
                  <p className="text-sm text-muted-foreground">Manage LLM providers and retrieval settings.</p>
                </div>

                <div className="grid gap-6">
                  <Suspense fallback={<LoadingSpinner message="Loading onboarding..." />}>
                    <OnboardingFlow
                      onboarding={onboarding}
                      apiReady={apiReady}
                      docsReady={docsReady}
                      modelsReady={modelsReady}
                      isSeedingDemo={isSeedingDemo}
                      activeMode={onboardingMode}
                      onModeChange={setOnboardingMode}
                      onSeedDemo={handleSeedDemo}
                      onOpenTab={setActiveTab}
                      onAskExample={handleAskExample}
                    />
                  </Suspense>

                  {/* Provider Carousel - Ollama, LM Studio, OpenRouter */}
                  <Suspense fallback={<LoadingSpinner message="Loading providers..." />}>
                    <ProviderCarousel
                      key={currentScope().generation}
                      config={config}
                      setConfig={setConfig}
                      isSavingConfig={isSavingConfig}
                      handleSaveConfig={handleSaveConfig}
                      persistConfig={persistConfig}
                      localProviders={localProviders}
                      localProvidersStatus={localProvidersStatus}
                      refreshLocalProviders={() => void refreshLocalProviders(currentScope())}
                      localSelections={localSelections}
                      setLocalSelection={setLocalSelection}
                      applyLocalProvider={applyLocalProvider}
                      apiBaseUrl={baseUrl}
                      apiKey={apiKey}
                      isConnected={isConnected}
                    />
                  </Suspense>

                  {/* Advanced Provider Configuration */}
                  <Suspense fallback={<LoadingSpinner message="Loading configuration..." />}>
                    <ProviderConfig
                      config={config}
                      setConfig={setConfig}
                      updateDeploymentProfile={updateDeploymentProfile}
                      updateBackend={updateBackend}
                      handleSaveConfig={handleSaveConfig}
                      modelOptions={modelOptions}
                      handleDiscoverModels={handleDiscoverModels}
                      isSavingConfig={isSavingConfig}
                      selectedProfile={selectedProfile}
                      handleSelectProfile={handleSelectProfile}
                      handleAddProfile={handleAddProfile}
                      newProfileName={newProfileName}
                      setNewProfileName={setNewProfileName}
                      refreshAll={() => void refreshAll(currentScope())}
                    />
                  </Suspense>

                  {/* Advanced Engineering Presets */}
                  <div className="bg-card rounded-lg border border-border/60 p-6 shadow-sm">
                    <div className="flex items-center gap-2 mb-2">
                      <Settings className="h-5 w-5 text-primary" />
                      <h3 className="font-semibold">Advanced Engineering Presets</h3>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">
                      Apply tuned defaults for common workflows
                    </p>
                    <Suspense fallback={<LoadingSpinner message="Loading presets..." />}>
                      <PresetSelector
                        value={activePreset}
                        onChange={async (newPreset: PresetLevel) => {
                          const scope = currentScope();
                          setActivePreset(newPreset);
                          try {
                            const updated = await fetchJson<AppConfig>(scope, `/config/presets/${newPreset}`, { method: "POST" });
                            if (!isCurrentScope(scope)) return;
                            setConfig(updated);
                            toast({ title: `Applied ${newPreset} preset`, variant: "default" });
                          } catch (e) {
                            if (!isCurrentScope(scope)) return;
                            toast({ title: toMessage(e), variant: "error" });
                          }
                        }}
                        disabled={isSavingConfig}
                      />
                    </Suspense>
                  </div>

                  {/* Retrieval Settings */}
                  <Suspense fallback={<LoadingSpinner message="Loading RAG settings..." />}>
                    <AdvancedRAGSettings
                      retrieval={config?.retrieval}
                      backends={config?.backends}
                      fallbacks={config?.fallbacks}
                      updateBackend={updateBackend}
                      updateRetrieval={updateRetrieval}
                      onSave={handleSaveConfig}
                      isSaving={isSavingConfig}
                      modelStatus={modelStatus}
                      onRefreshModelStatus={() => refreshModelStatus(currentScope(), config?.retrieval?.embedding_model, config?.retrieval?.reranker_model)}
                      onDownloadEmbedding={() => downloadModel("embedding")}
                      onDownloadReranker={() => downloadModel("reranker")}
                      onDeleteEmbedding={() => deleteModel("embedding")}
                      onDeleteReranker={() => deleteModel("reranker")}
                      isDownloadingEmbedding={isDownloadingEmbedding}
                      isDownloadingReranker={isDownloadingReranker}
                      isCheckingModels={isCheckingModels}
                      modelActionMessage={modelActionMessage}
                    />
                  </Suspense>

                  {/* Cache Controls */}
                  <div className="bg-card rounded-lg border border-border/60 p-6 shadow-sm">
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <BarChart3 className="h-5 w-5 text-primary" />
                        <h3 className="font-semibold">System Cache</h3>
                      </div>
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={handleClearCache}
                        disabled={isClearingCache}
                      >
                        {isClearingCache ? "Clearing..." : "Clear All Caches"}
                      </Button>
                    </div>
                    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 text-sm">
                      <div className="p-3 bg-muted/40 rounded-lg border border-border/40">
                        <div className="text-muted-foreground mb-1">Embedding Cache</div>
                        <div className="text-xl font-mono font-medium">{cacheStats?.embeddings?.size || 0}</div>
                      </div>
                      <div className="p-3 bg-muted/40 rounded-lg border border-border/40">
                        <div className="text-muted-foreground mb-1">Query Cache</div>
                        <div className="text-xl font-mono font-medium">{cacheStats?.queries?.size || 0}</div>
                      </div>
                      <div className="p-3 bg-muted/40 rounded-lg border border-border/40">
                        <div className="text-muted-foreground mb-1">Cache Size</div>
                        <div className="text-xl font-mono font-medium">{(cacheStats?.embeddings?.size || 0) + (cacheStats?.queries?.size || 0)} items</div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div id="tabpanel-documents" role="tabpanel" aria-labelledby="tab-documents" tabIndex={0} className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "documents" ? "hidden" : ""}`}>
              {/* Documents Panel */}
              <div className="space-y-6 max-w-[1600px] mx-auto flex flex-col">
                <div>
                  <h2 className="text-lg font-semibold tracking-tight">Knowledge Base</h2>
                  <p className="text-sm text-muted-foreground">Ingest and manage documents.</p>
                </div>

                <div className="grid lg:grid-cols-[350px,1fr] gap-6 min-h-0">
                  <Suspense fallback={<LoadingSpinner message="Loading ingest panel..." />}>
                    <IngestPanel
                      ingestTitle={ingestTitle}
                      setIngestTitle={setIngestTitle}
                      ingestText={ingestText}
                      setIngestText={setIngestText}
                      handleIngest={handleIngest}
                      isIngesting={isIngesting}
                      ingestSync={ingestSync}
                      setIngestSync={setIngestSync}
                      startUploadBatch={startUploadBatch}
                      isUploadingFile={isUploadingFile}
                      fileInputRef={fileInputRef}
                      documents={documents}
                      handleDeleteDocument={handleDeleteDocument}
                      handleDeleteAllDocuments={handleDeleteAllDocuments}
                      formatDateTime={formatDateTime}
                      langextractProfileOverride={langextractProfileOverride}
                      setLangextractProfileOverride={setLangextractProfileOverride}
                      langextractPromptOverride={langextractPromptOverride}
                      setLangextractPromptOverride={setLangextractPromptOverride}
                      langextractDefaultProfile={config?.retrieval?.langextract_profile_default}
                      ocrPolicy={config?.ingest?.ocr?.policy ?? "auto"}
                      setOcrPolicy={updateIngestOcrPolicy}
                    />
                  </Suspense>

                  <div className="bg-card rounded-lg border border-border/60 shadow-sm flex flex-col min-h-0">
                    <div className="p-4 border-b border-border/40 flex items-center justify-between">
                      <h3 className="font-semibold flex items-center gap-2">
                        <FileText className="h-4 w-4 text-primary" />
                        Documents ({documents.length})
                      </h3>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="text-destructive hover:text-destructive hover:bg-destructive/10 h-8"
                        onClick={handleDeleteAllDocuments}
                        disabled={documents.length === 0}
                      >
                        Delete All
                      </Button>
                    </div>

                    <div className="flex-1 overflow-auto p-2">
                      {documents.length === 0 ? (
                        <div className="h-full flex flex-col items-center justify-center text-muted-foreground p-8">
                          <FileText className="h-12 w-12 mb-4 opacity-20" />
                          <p>No documents found.</p>
                          <p className="text-xs opacity-70">Add content using the ingest panel.</p>
                        </div>
                      ) : (
                        <div className="space-y-2">
                          {documents.map((doc) => (
                            <div key={doc.id} className="group flex items-center justify-between p-3 rounded-lg border border-border/40 bg-background hover:border-primary/40 transition-all">
                              <div>
                                <div className="font-medium text-sm truncate flex-1 min-w-0">{doc.title}</div>
                                <div className="flex items-center gap-2 text-xs text-muted-foreground mt-1">
                                  <span className="bg-muted px-1.5 py-0.5 rounded text-[10px] uppercase">{doc.metadata?.source || "text"}</span>
                                  <span>{formatDateTime(doc.created_at)}</span>
                                  <span>- {doc.chunk_count} chunks</span>
                                </div>
                              </div>
                              <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity">
                                <div className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${doc.metadata?.processing_status === 'ready' ? 'bg-primary/10 text-primary' :
                                  doc.metadata?.processing_status === 'error' ? 'bg-destructive/10 text-destructive' :
                                    'bg-muted text-muted-foreground'
                                  }`}>
                                  {doc.metadata?.processing_status || 'unknown'}
                                </div>
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  className="h-8 w-8 text-muted-foreground hover:text-destructive"
                                  onClick={() => handleDeleteDocument(doc.id, doc.title)}
                                >
                                  <span className="sr-only">Delete</span>
                                  <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 6h18" /><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" /><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" /></svg>
                                </Button>
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div id="tabpanel-query" role="tabpanel" aria-labelledby="tab-query" tabIndex={0} className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "query" ? "hidden" : ""}`}>
              {/* Query Interface */}
              <Suspense fallback={<LoadingSpinner message="Loading chat interface..." />}>
                <ChatInterface
                  question={question}
                  setQuestion={setQuestion}
                  handleAsk={handleAsk}
                  isQuerying={isQuerying}
                  queryResult={queryResult}
                  history={chatHistory}
                  documents={documents}
                  selectedDocumentIds={selectedDocumentIds}
                  setSelectedDocumentIds={setSelectedDocumentIds}
                  activeStage={activeStage}
                  progress={progress ? { ...progress, progress: progress.progress || 0 } : undefined}
                  providerConfig={config?.provider}
                  baseUrl={baseUrl}
                  apiKey={apiKey}
                  onNewChat={handleNewChat}
                  onCancel={handleCancelQuery}
                  savedSessions={savedSessions}
                  onLoadSession={handleLoadSession}
                  onDeleteSession={handleDeleteSession}
                  onClearHistory={handleClearHistory}
                  scrollRef={chatEndRef}
                  currentSessionId={currentSessionId}
                  preset={activePreset}
                  queryMode={queryMode}
                  onQueryModeChange={setQueryMode}
                  onPresetChange={async (preset) => {
                    const scope = currentScope();
                    setActivePreset(preset);
                    try {
                      const updated = await fetchJson<AppConfig>(scope, `/config/presets/${preset}`, { method: "POST" });
                      if (!isCurrentScope(scope)) return;
                      setConfig(updated);
                      toast({ title: "Preset applied", description: `Switched to ${preset} mode`, variant: "success" });
                    } catch (e) {
                      console.error("Failed to apply preset:", e);
                    }
                  }}
                />
              </Suspense>
            </div>

            <div id="tabpanel-quality" role="tabpanel" aria-labelledby="tab-quality" tabIndex={0} className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "quality" ? "hidden" : ""}`}>
              <div className="mx-auto max-w-[1400px]">
                <Suspense fallback={<LoadingSpinner message="Loading quality cockpit..." />}>
                  <QualityCockpit
                    key={currentScope().generation}
                    documents={documents}
                    buildUrl={buildUrl}
                    buildHeaders={buildHeaders}
                    onPresetPromoted={() => void refreshAll(currentScope())}
                  />
                </Suspense>
              </div>
            </div>

            <div id="tabpanel-metrics" role="tabpanel" aria-labelledby="tab-metrics" tabIndex={0} className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "metrics" ? "hidden" : ""}`}>
              {/* Metrics & Traces */}
              <div className="space-y-6 max-w-5xl mx-auto">
                <Suspense fallback={<LoadingSpinner message="Loading status panel..." />}>
                  <EnterpriseStatusPanel key={currentScope().generation} baseUrl={baseUrl} apiKey={apiKey} />
                </Suspense>

                <Suspense fallback={<LoadingSpinner message="Loading metrics..." />}>
                  <MetricsDashboard
                    traces={traces}
                    cacheStats={cacheStats ?? undefined}
                    onClearCache={handleClearCache}
                    isClearingCache={isClearingCache}
                  />
                </Suspense>
                <Suspense fallback={<LoadingSpinner message="Loading traces..." />}>
                  <TraceLog
                    isEvaluating={isEvaluating}
                    handleEvaluation={handleEvaluation}
                    evaluationSummary={evaluationSummary}
                    traces={traces}
                    formatNumber={formatNumber}
                  />
                </Suspense>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

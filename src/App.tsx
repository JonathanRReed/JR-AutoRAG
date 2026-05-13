import { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
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
import { buildApiUrl, resolveDefaultApiBaseUrl } from "@/lib/api-url";

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
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [apiKey, setApiKey] = useState(() => {
    if (typeof window === "undefined") {
      return "";
    }
    return sessionStorage.getItem("jr-autorag-api-key") || "";
  });
  const [status, setStatus] = useState("");
  const [isConnected, setIsConnected] = useState(false); // New state for reliable connection tracking
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
  const headers = useMemo(() => buildHeaders({ "Content-Type": "application/json" }), [buildHeaders]);
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
    if (trimmed) {
      sessionStorage.setItem("jr-autorag-api-key", trimmed);
    } else {
      sessionStorage.removeItem("jr-autorag-api-key");
    }
  }, [apiKey]);

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

  const buildUrl = (path: string) => buildApiUrl(baseUrl, path);

  const fetchJson = async <T,>(path: string, init?: RequestInit): Promise<T> => {
    try {
      const response = await fetch(buildUrl(path), {
        ...init,
        headers: buildHeaders(init?.headers),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      setIsConnected(true); // Successfully reached API
      return response.json();
    } catch (error) {
      if (error instanceof TypeError) {
        setIsConnected(false); // Network error
      }
      throw error;
    }
  };

  // Heartbeat for connection status
  useEffect(() => {
    const check = async () => {
      try {
        await fetch(buildUrl("/healthz"), { method: "GET", headers: buildHeaders() });
        setIsConnected(true);
      } catch (e) {
        setIsConnected(false);
      }
    };

    // Initial check
    check();

    // Periodically check every 20 seconds
    const id = setInterval(check, 20000);
    return () => clearInterval(id);
  }, [baseUrl, buildHeaders]);

  const refreshLocalProviders = async () => {
    setLocalProvidersStatus("loading");
    try {
      const providers = await fetchJson<LocalProviderInfo[]>("/providers/local");
      setLocalProviders(providers);
      setLocalProvidersStatus("ready");
      if (!providers.length) {
        setStatus("No local runtimes detected. Launch Ollama or LM Studio and rescan.");
      }
    } catch (error) {
      setLocalProvidersStatus("error");
      setStatus(`Local provider scan failed: ${toMessage(error)}`);
    }
  };

  const waitForDocumentReady = async (documentId: string, label?: string) => {
    const timeoutMs = 120_000;
    const intervalMs = 1500;
    const deadline = Date.now() + timeoutMs;
    let lastDocs: DocumentOut[] = [];

    while (Date.now() < deadline) {
      const docs = await fetchJson<DocumentOut[]>("/documents");
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
    try {
      const response = await fetch(buildUrl(`/documents/${id}`), { method: "DELETE", headers: buildHeaders() });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      setStatus(`Deleted ${title}`);
      toast({ title: "Document deleted", description: title, variant: "success" });
      refreshAll();
    } catch (error) {
      setStatus(`Delete failed: ${toMessage(error)}`);
      toast({ title: "Delete failed", description: toMessage(error), variant: "error" });
    }
  };

  const handleDeleteAllDocuments = async () => {
    if (!window.confirm("Are you sure you want to clear ALL ingested documents? This cannot be undone.")) {
      return;
    }
    setStatus("Clearing knowledge base...");
    try {
      const response = await fetch(buildUrl("/documents"), {
        method: "DELETE",
        headers,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      setStatus("Knowledge base cleared");
      toast({ title: "Knowledge base cleared", description: "All documents deleted", variant: "success" });
      refreshAll();
    } catch (error) {
      setStatus(`Clear failed: ${toMessage(error)}`);
      toast({ title: "Clear failed", description: toMessage(error), variant: "error" });
    }
  };

  const refreshAll = async () => {
    try {
      const [cfg, docs, traceList, cache, posture] = await Promise.all([
        fetchJson<AppConfig>("/config"),
        fetchJson<DocumentOut[]>("/documents"),
        fetchJson<TraceOut[]>("/monitoring/traces"),
        fetchJson<CacheStats>("/monitoring/cache"),
        fetchJson<SecurityPostureResponse>("/security/posture"),
      ]);
      setConfig(cfg);
      setSelectedProfile(cfg.profile);
      setNewProfileName(cfg.profile);
      setDocuments(docs);
      setTraces(traceList);
      setCacheStats(cache);
      setSecurityPosture(posture);
      setStatus("API data loaded");
      setIsConnected(true);
    } catch (error) {
      setStatus(`Failed to load data: ${toMessage(error)}`);
    }
  };

  const refreshOnboarding = async () => {
    try {
      const data = await fetchJson<OnboardingState>("/onboarding");
      setOnboarding(data);
    } catch (error) {
      setStatus(`Onboarding failed to load: ${toMessage(error)}`);
    }
  };

  const refreshModelStatus = async (embeddingModel?: string, rerankerModel?: string) => {
    if (!embeddingModel && !rerankerModel) {
      return;
    }
    setIsCheckingModels(true);
    try {
      const status = await fetchJson<ModelStatus>("/config/models/status", {
        method: "POST",
        headers,
        body: JSON.stringify({
          embedding_model: embeddingModel,
          reranker_model: rerankerModel,
        }),
      });
      setModelStatus(status);
    } catch (error) {
      setModelStatus({
        embedding: "error",
        reranker: "error",
        embedding_message: toMessage(error),
        reranker_message: toMessage(error),
      });
    } finally {
      setIsCheckingModels(false);
    }
  };

  const downloadInstallReport = async () => {
    setIsExportingInstallReport(true);
    try {
      const report = await fetchJson<InstallReportResponse>("/install/report");
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
      setStatus(`Install report export failed: ${toMessage(error)}`);
      toast({ title: "Install report export failed", description: toMessage(error), variant: "error" });
    } finally {
      setIsExportingInstallReport(false);
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
    if (kind === "embedding") {
      setIsDownloadingEmbedding(true);
    } else {
      setIsDownloadingReranker(true);
    }
    try {
      await fetchJson("/config/models/download", {
        method: "POST",
        headers,
        body: JSON.stringify({ kind, model }),
      });
      const message = `Downloaded ${kind} model`;
      setStatus(message);
      setModelActionMessage(message);
      await refreshModelStatus(
        config.retrieval.embedding_model,
        config.retrieval.reranker_model,
      );
    } catch (error) {
      const message = `Download failed: ${toMessage(error)}`;
      setStatus(message);
      setModelActionMessage(message);
    } finally {
      if (kind === "embedding") {
        setIsDownloadingEmbedding(false);
      } else {
        setIsDownloadingReranker(false);
      }
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
    try {
      await fetchJson("/config/models/delete", {
        method: "POST",
        headers,
        body: JSON.stringify({ kind, model }),
      });
      const message = `Removed ${kind} model cache`;
      setStatus(message);
      setModelActionMessage(message);
      await refreshModelStatus(
        config.retrieval.embedding_model,
        config.retrieval.reranker_model,
      );
    } catch (error) {
      const message = `Remove failed: ${toMessage(error)}`;
      setStatus(message);
      setModelActionMessage(message);
    }
  };

  useEffect(() => {
    refreshAll();
    refreshOnboarding();
    refreshLocalProviders();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseUrl]);

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
      config.retrieval.embedding_model,
      config.retrieval.reranker_model,
    );
  }, [config?.retrieval?.embedding_model, config?.retrieval?.reranker_model]);

  const handleTestConnection = async () => {
    setStatus("Testing connection...");
    try {
      await fetchJson("/healthz");
      setStatus("API reachable");
      setIsConnected(true);
      toast({ title: "API reachable", description: baseUrl, variant: "success" });
    } catch (error) {
      setStatus(`Health check failed: ${toMessage(error)}`);
      setIsConnected(false);
      toast({ title: "API health check failed", description: toMessage(error), variant: "error" });
    }
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

  const persistConfig = async (nextConfig: AppConfig, message = "Configuration saved") => {
    setIsSavingConfig(true);
    try {
      const updated = await fetchJson<AppConfig>("/config", {
        method: "PUT",
        headers,
        body: JSON.stringify({ ...nextConfig, profile: selectedProfile }),
      });
      setConfig(updated);
      setStatus(message);
      toast({ title: "Configuration saved", description: nextConfig.provider?.name || "Provider updated", variant: "success" });
    } catch (error) {
      console.error("Error saving configuration:", error);
      setStatus(`Save failed: ${toMessage(error)}`);
      toast({ title: "Save failed", description: toMessage(error), variant: "error" });
      throw error;
    } finally {
      setIsSavingConfig(false);
    }
  };

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
    setStatus("Discovering models...");
    try {
      const models = await fetchJson<string[]>("/config/models", {
        method: "POST",
        headers,
        body: JSON.stringify(config.provider),
      });
      setModelOptions(models);
      setStatus(`Found ${models.length} models`);
      toast({ title: "Models discovered", description: `${models.length} options found`, variant: "success" });
    } catch (error) {
      console.error("Error discovering models:", error);
      setStatus(`Model discovery failed: ${toMessage(error)}`);
      toast({ title: "Model discovery failed", description: toMessage(error), variant: "error" });
    }
  };

  const handleIngest = async () => {
    setIsIngesting(true);
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
      const result = await fetchJson<IngestResponse>("/documents/text", {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      setStatus(`Processing ${result.title}...`);
      setIngestText("");
      await waitForDocumentReady(result.document_id, result.title);
      toast({ title: "Ingest queued", description: result.title, variant: "success" });
    } catch (error) {
      setStatus(`Ingest failed: ${toMessage(error)}`);
      toast({ title: "Ingest failed", description: toMessage(error), variant: "error" });
    } finally {
      setIsIngesting(false);
    }
  };

  const handleAsk = async (overrideQuestion?: string, isRegenerate = false) => {
    if (isQuerying) return; // Prevent double trigger

    // 1. Determine the question to ask
    const currentQuestion = overrideQuestion || question;
    if (!currentQuestion.trim()) return;

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
      const response = await fetch(buildUrl("/query/stream"), {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      if (!response.ok || !response.body) {
        throw new Error(await response.text());
      }

      setIsConnected(true); // Stream started

      const body = response.body;
      if (!body) {
        throw new Error("Response body is null");
      }

      const reader = body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let receivedResult = false;

      while (true) {
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
            if (localSessionId === currentSessionIdRef.current) {
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
              if (localSessionId === currentSessionIdRef.current) {
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
            if (localSessionId === currentSessionIdRef.current) {
              setActiveStage(event.data?.name ?? null);
            }
          }
          if (event.type === "progress") {
            if (localSessionId === currentSessionIdRef.current) {
              setProgress(event.data);
              if (event.data?.trace_id) {
                setCurrentTraceId(event.data.trace_id);
              }
            }
          }

          if (event.type === "result") {
            receivedResult = true;
            if (localSessionId === currentSessionIdRef.current) {
              setQueryResult(event.data);
              setStatus("Query succeeded");
              setActiveStage(null);
              setProgress(null);
            }

            // Update history locally
            const updatedHistory = [
              ...initialHistory,
              { role: "assistant", content: event.data.answer }
            ];

            if (localSessionId === currentSessionIdRef.current) {
              setChatHistory(updatedHistory);
            }

            // ALWAYS update history in the saved session list (even if not active)
            setSavedSessions(prev => prev.map(s =>
              s.id === localSessionId ? { ...s, history: updatedHistory, queryResult: event.data } : s
            ));

            refreshAll();
            toast({ title: "Query complete", description: "Results updated", variant: "success" });
          }
          if (event.type === "error") {
            const message = event.data?.message ?? "Query failed";
            throw new Error(message);
          }
        }
      }

      if (!receivedResult) {
        throw new Error("Query ended before returning a result");
      }
    } catch (error) {
      console.error("Query execution error:", error);
      setStatus(`Query failed: ${toMessage(error)}`);

      // Attempt to save partial result if available
      setQueryResult((currentPartial) => {
        if (currentPartial?.answer && localSessionId === currentSessionIdRef.current) {
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
      setIsQuerying(false);
      setActiveStage(null);
      setProgress(null);
      setCurrentTraceId(null);
    }
  };

  const handleCancelQuery = async () => {
    if (!currentTraceId) return;
    try {
      await fetch(buildUrl(`/query/cancel?trace_id=${currentTraceId}`), { method: "POST", headers: buildHeaders() });
      setStatus("Query cancelled");
      toast({ title: "Query cancelled", variant: "info" });
      setIsQuerying(false);
      setActiveStage(null);
      setProgress(null);
      setCurrentTraceId(null);
    } catch (e) {
      console.error("Cancel failed", e);
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
    setIsEvaluating(true);
    try {
      const payload = {
        name: "Smoke Test",
        questions: ["What is JR AutoRAG?", "How do I onboard documents?"],
      };
      const result = await fetchJson<{ average_coverage: number; average_tokens: number }>("/evaluation", {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      setEvaluationSummary(
        `Avg coverage ${(result.average_coverage * 100).toFixed(1)}%, Avg tokens ${result.average_tokens.toFixed(0)}`,
      );
      setStatus("Evaluation complete");
      toast({ title: "Evaluation complete", description: "Coverage and tokens updated", variant: "success" });
    } catch (error) {
      const message = `Evaluation failed: ${toMessage(error)}`;
      setEvaluationSummary(message);
      setStatus(message);
      toast({ title: "Evaluation failed", description: toMessage(error), variant: "error" });
    } finally {
      setIsEvaluating(false);
    }
  };

  const handleSeedDemo = async () => {
    setIsSeedingDemo(true);
    setStatus("Loading demo corpus...");
    try {
      const result = await fetchJson<DemoSeedResponse>("/onboarding/demo/seed", {
        method: "POST",
        headers,
      });
      setStatus(`Demo corpus ready with ${result.document_count} document(s)`);
      toast({
        title: "Demo corpus ready",
        description: `${result.seeded.length} added, ${result.skipped.length} already present`,
        variant: "success",
      });
      await refreshAll();
      await refreshOnboarding();
    } catch (error) {
      setStatus(`Demo setup failed: ${toMessage(error)}`);
      toast({ title: "Demo setup failed", description: toMessage(error), variant: "error" });
    } finally {
      setIsSeedingDemo(false);
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
    setIsClearingCache(true);
    try {
      const result = await fetchJson<{ status: string; message: string }>("/api/cache/clear", {
        method: "DELETE",
      });
      setStatus(result.message || "Cache cleared");
      refreshAll();
      toast({ title: "Cache cleared", description: "All indexes and query caches reset", variant: "success" });
    } catch (error) {
      setStatus(`Cache clear failed: ${toMessage(error)}`);
      toast({ title: "Cache clear failed", description: toMessage(error), variant: "error" });
    } finally {
      setIsClearingCache(false);
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
    const selection = (localSelections[provider.base_url] || {}) as Partial<RoleSelection>;
    const ensureConfig = async () => {
      if (config) {
        return config;
      }
      setStatus("Loading configuration...");
      try {
        const cfg = await fetchJson<AppConfig>("/config");
        setConfig(cfg);
        setSelectedProfile(cfg.profile);
        setNewProfileName(cfg.profile);
        return cfg;
      } catch (error) {
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
    setConfig(nextConfig);
    try {
      await persistConfig(nextConfig, `Applied ${provider.name} (${planner}/${generator})`);
      setStatus(`Successfully applied ${provider.name} with ${planner}`);
      toast({ title: "Provider applied", description: `${provider.name} (${planner}/${generator})`, variant: "success" });
    } catch {
      // status already updated inside persistConfig
    }
  };

  const uploadFile = async (file: File): Promise<IngestResponse> => {
    setIsUploadingFile(true);
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
      const resp = await fetch(buildUrl("/documents/upload"), {
        method: "POST",
        headers: buildHeaders(),
        body: formData,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || "Upload failed");
      }
      const result: IngestResponse = await resp.json();
      setStatus(`Processing ${file.name}...`);
      setIngestTitle("");
      setIngestText("");
      return result;
    } catch (error) {
      setStatus(`Upload failed: ${toMessage(error)}`);
      toast({ title: "Upload failed", description: toMessage(error), variant: "error" });
      throw error;
    } finally {
      setIsUploadingFile(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
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
  const apiReady = isConnected || Boolean(config);
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

          {/* Top Navigation */}
          <nav className="no-scrollbar flex min-w-0 items-center gap-1 overflow-x-auto rounded-lg bg-muted/30 p-1" role="tablist" aria-label="Main navigation">
            {tabs.map(tab => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
	                  role="tab"
	                  aria-selected={isActive}
	                  aria-controls={`tabpanel-${tab.id}`}
	                  aria-label={tab.label}
	                  title={tab.label}
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
            {/* API URL */}
            <form
              className="hidden 2xl:flex items-center gap-2"
              onSubmit={(event) => {
                event.preventDefault();
                void handleTestConnection();
              }}
            >
              <Input
                className="w-48 text-xs"
                value={baseUrl}
                onChange={e => setBaseUrl(e.target.value)}
                placeholder="API URL"
                aria-label="API base URL"
              />
              <Input
                className="w-40 text-xs"
                value={apiKey}
                onChange={e => setApiKey(e.target.value)}
                type="password"
                autoComplete="off"
                placeholder="X-API-Key (session)"
                aria-label="API key"
              />
              <Button size="sm" variant="outline" type="submit">
                {apiReady ? "Connected" : "Connect"}
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
            <div className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "home" ? "hidden" : ""}`}>
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
                        <FileText className="size-4" />
                        Ingest Client Documents
                      </Button>
                      <div className="grid grid-cols-2 gap-2">
                        <Button variant="outline" onClick={() => setActiveTab("config")}>
                          <Settings className="size-4" />
                          Configure
                        </Button>
                        <Button variant="outline" onClick={() => setActiveTab("query")}>
                          <MessageSquare className="size-4" />
                          Query
                        </Button>
                      </div>
                      <Button variant="secondary" onClick={() => setActiveTab("quality")}>
                        <FileCheck2 className="size-4" />
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
                            <ArrowRight className="size-4" />
                          </Button>
                        </div>
                        <div className="flex items-center justify-between gap-3 rounded-md border border-border/50 bg-muted/10 px-3 py-2">
                          <span className="text-muted-foreground">Quality receipts</span>
                          <Button variant="ghost" size="sm" onClick={() => setActiveTab("quality")}>
                            Open
                            <ArrowRight className="size-4" />
                          </Button>
                        </div>
                        <div className="flex items-center justify-between gap-3 rounded-md border border-border/50 bg-muted/10 px-3 py-2">
                          <span className="text-muted-foreground">Runtime traces</span>
                          <Button variant="ghost" size="sm" onClick={() => setActiveTab("metrics")}>
                            Inspect
                            <ArrowRight className="size-4" />
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
                          <ArrowRight className="size-4" />
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

            <div className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "config" ? "hidden" : ""}`}>
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
                      config={config}
                      setConfig={setConfig}
                      isSavingConfig={isSavingConfig}
                      handleSaveConfig={handleSaveConfig}
                      persistConfig={persistConfig}
                      localProviders={localProviders}
                      localProvidersStatus={localProvidersStatus}
                      refreshLocalProviders={refreshLocalProviders}
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
                      refreshAll={refreshAll}
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
                          setActivePreset(newPreset);
                          try {
                            const updated = await fetchJson<AppConfig>(`/config/presets/${newPreset}`, { method: "POST" });
                            setConfig(updated);
                            toast({ title: `Applied ${newPreset} preset`, variant: "default" });
                          } catch (e) {
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
                      onRefreshModelStatus={() => refreshModelStatus(config?.retrieval?.embedding_model, config?.retrieval?.reranker_model)}
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

            <div className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "documents" ? "hidden" : ""}`}>
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
                      uploadFile={uploadFile}
                      isUploadingFile={isUploadingFile}
                      fileInputRef={fileInputRef}
                      documents={documents}
                      handleDeleteDocument={handleDeleteDocument}
                      handleDeleteAllDocuments={handleDeleteAllDocuments}
                      waitForDocumentReady={waitForDocumentReady}
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
                              <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
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

            <div className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "query" ? "hidden" : ""}`}>
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
                    setActivePreset(preset);
                    try {
                      const updated = await fetchJson<AppConfig>(`/config/presets/${preset}`, { method: "POST" });
                      setConfig(updated);
                      toast({ title: "Preset applied", description: `Switched to ${preset} mode`, variant: "success" });
                    } catch (e) {
                      console.error("Failed to apply preset:", e);
                    }
                  }}
                />
              </Suspense>
            </div>

            <div className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "quality" ? "hidden" : ""}`}>
              <div className="mx-auto max-w-[1400px]">
                <Suspense fallback={<LoadingSpinner message="Loading quality cockpit..." />}>
                  <QualityCockpit
                    documents={documents}
                    buildUrl={buildUrl}
                    buildHeaders={buildHeaders}
                    onPresetPromoted={refreshAll}
                  />
                </Suspense>
              </div>
            </div>

            <div className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "metrics" ? "hidden" : ""}`}>
              {/* Metrics & Traces */}
              <div className="space-y-6 max-w-5xl mx-auto">
                <Suspense fallback={<LoadingSpinner message="Loading status panel..." />}>
                  <EnterpriseStatusPanel baseUrl={baseUrl} apiKey={apiKey} />
                </Suspense>

                <Suspense fallback={<LoadingSpinner message="Loading metrics..." />}>
                  <MetricsDashboard traces={traces} />
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

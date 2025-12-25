import { useEffect, useMemo, useRef, useState } from "react";
import { BarChart3, FileText, MessageSquare, Moon, Settings, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/toast";

import { AdvancedRAGSettings } from "@/components/features/AdvancedRAGSettings";
import { ChatInterface } from "@/components/features/ChatInterface";
import { EnterpriseStatusPanel } from "@/components/features/EnterpriseStatusPanel";
import { IngestPanel } from "@/components/features/IngestPanel";
import { MetricsDashboard } from "@/components/features/MetricsDashboard";
import { ProviderConfig } from "@/components/features/ProviderConfig";
import { ProviderCarousel } from "@/components/features/ProviderCarousel";
import { TraceLog } from "@/components/features/TraceLog";
import { PresetSelector } from "@/components/features/PresetSelector";

import "./index.css";
import type { AppConfig, CacheStats, DocumentOut, IngestResponse, LocalProviderInfo, ModelStatus, ProviderConfig as ProviderConfigType, ProviderProfile, QueryResponse, RetrievalDefaults, RoleSelection, TraceOut, ChatSession, PresetLevel } from "@/types";

const resolveDefaultBaseUrl = () => {
  const envBase =
    (import.meta.env?.BUN_PUBLIC_API_BASE_URL as string | undefined) ||
    (import.meta.env?.VITE_API_BASE_URL as string | undefined);
  if (envBase) {
    return envBase;
  }
  if (typeof window !== "undefined") {
    try {
      const url = new URL(window.location.href);
      if (url.port === "3000") {
        url.port = "8000";
      }
      return `${url.protocol}//${url.hostname}${url.port ? `:${url.port}` : ""}`;
    } catch {
      return "http://localhost:8000";
    }
  }
  return "http://localhost:8000";
};

const defaultBaseUrl = resolveDefaultBaseUrl();

const formatNumber = (value?: number) =>
  typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "0.00";

const toMessage = (error: unknown) => {
  if (error instanceof Error) {
    return error.message;
  }
  return String(error);
};

const formatDateTime = (value?: string) => {
  if (!value) return "Unknown";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }
  return date.toLocaleString();
};

type TabId = "config" | "documents" | "query" | "metrics";

const tabs: { id: TabId; label: string; icon: typeof Settings }[] = [
  { id: "config", label: "Configuration", icon: Settings },
  { id: "documents", label: "Documents", icon: FileText },
  { id: "query", label: "Query", icon: MessageSquare },
  { id: "metrics", label: "Metrics", icon: BarChart3 },
];

export function App() {
  const { toast } = useToast();
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [status, setStatus] = useState("");
  const [isConnected, setIsConnected] = useState(false); // New state for reliable connection tracking
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [traces, setTraces] = useState<TraceOut[]>([]);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [question, setQuestion] = useState("");
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [chatHistory, setChatHistory] = useState<{ role: string; content: string }[]>(() => {
    const saved = localStorage.getItem("chatHistory");
    return saved ? JSON.parse(saved) : [];
  });
  const [savedSessions, setSavedSessions] = useState<ChatSession[]>(() => {
    const saved = localStorage.getItem("savedSessions");
    return saved ? JSON.parse(saved) : [];
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
  const [ingestSync, setIngestSync] = useState(true);
  const [activeTab, setActiveTab] = useState<TabId>(() => {
    if (typeof window === "undefined") return "config";
    const saved = localStorage.getItem("activeTab");
    if (saved && ["config", "documents", "query", "metrics"].includes(saved)) {
      return saved as TabId;
    }
    return "config";
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
  const [isDownloadingEmbedding, setIsDownloadingEmbedding] = useState(false);
  const [isDownloadingReranker, setIsDownloadingReranker] = useState(false);
  const [modelActionMessage, setModelActionMessage] = useState("");
  const [isDarkMode, setIsDarkMode] = useState(() => {
    if (typeof window !== "undefined") {
      return window.matchMedia("(prefers-color-scheme: dark)").matches;
    }
    return false;
  });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const headers = useMemo(() => ({ "Content-Type": "application/json" }), []);
  const [activePreset, setActivePreset] = useState<PresetLevel>(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("jr-autorag-preset") as PresetLevel | null;
      if (saved && ["turbo", "fast", "balanced", "thorough", "ultra_accurate"].includes(saved)) {
        return saved;
      }
    }
    return "balanced";
  });

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
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDarkMode]);

  const chatEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (activeTab === "query") {
      chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }
  }, [chatHistory, queryResult?.answer, activeTab]);

  const buildUrl = (path: string) => `${baseUrl.replace(/\/$/, "")}${path}`;

  const fetchJson = async <T,>(path: string, init?: RequestInit): Promise<T> => {
    try {
      const response = await fetch(buildUrl(path), init);
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
        await fetch(buildUrl("/healthz"), { method: "GET" });
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
  }, [baseUrl]);

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
      await fetch(buildUrl(`/documents/${id}`), { method: "DELETE" });
      setDocuments(prev => prev.filter(doc => doc.id !== id));
      setStatus(`Deleted ${title}`);
    } catch (error) {
      setStatus(`Delete failed: ${toMessage(error)}`);
    }
  };

  const handleDeleteAllDocuments = async () => {
    if (!window.confirm("Are you sure you want to clear ALL ingested documents? This cannot be undone.")) {
      return;
    }
    setStatus("Clearing knowledge base...");
    try {
      await fetch(buildUrl("/documents"), {
        method: "DELETE",
        headers,
      });
      setStatus("Knowledge base cleared");
      refreshAll();
    } catch (error) {
      setStatus(`Clear failed: ${toMessage(error)}`);
    }
  };

  const refreshAll = async () => {
    try {
      const [cfg, docs, traceList, cache] = await Promise.all([
        fetchJson<AppConfig>("/config"),
        fetchJson<DocumentOut[]>("/documents"),
        fetchJson<TraceOut[]>("/monitoring/traces"),
        fetchJson<CacheStats>("/monitoring/cache"),
      ]);
      setConfig(cfg);
      setSelectedProfile(cfg.profile);
      setNewProfileName(cfg.profile);
      setDocuments(docs);
      setTraces(traceList);
      setCacheStats(cache);
      setStatus("API data loaded");
      setIsConnected(true);
    } catch (error) {
      setStatus(`Failed to load data: ${toMessage(error)}`);
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
    refreshLocalProviders();
    handleTestConnection(); // Check connection explicitly on mount/change
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
      const result = await fetchJson<IngestResponse>("/documents/text", {
        method: "POST",
        headers,
        body: JSON.stringify({ title: ingestTitle, text: ingestText, sync: ingestSync }),
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
        ? { question: currentQuestion, document_ids: selectedDocumentIds, history: historyToSend }
        : { question: currentQuestion, history: historyToSend };
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
          const event = JSON.parse(payloadText);
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
      await fetch(buildUrl(`/query/cancel?trace_id=${currentTraceId}`), { method: "POST" });
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
      const resp = await fetch(buildUrl("/documents/upload"), {
        method: "POST",
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

  const docsReady = documents.length > 0;
  const modelsReady = Boolean(config?.provider?.planner_model && config?.provider?.generator_model);
  const apiReady = isConnected;

  return (
    <div className="flex flex-col h-screen overflow-hidden bg-background transition-colors duration-300">
      {/* Header */}
      {/* Header */}
      <header className="flex-none relative z-50 border-b border-border/60 bg-background/95">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-4 py-3">
          <div className="flex items-center gap-8">
            <div className="flex items-center gap-3">
              <img
                src="/HWC-Icon.png"
                alt="JR AutoRAG"
                className="h-10 w-10 rounded-lg object-cover shadow-sm"
              />
              <div>
                <h1 className="text-xl font-bold text-foreground">JR AutoRAG</h1>
                <p className="text-xs text-muted-foreground">Admin Console</p>
              </div>
            </div>

            {/* Top Navigation */}
            <nav className="flex items-center gap-1 bg-muted/30 p-1 rounded-lg">
              {tabs.map(tab => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={`flex items-center gap-2 rounded-md px-3 py-1.5 text-sm font-medium transition-all ${isActive
                      ? "bg-primary text-primary-foreground shadow-sm"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                      }`}
                  >
                    <Icon className="h-4 w-4" />
                    {tab.label}
                  </button>
                );
              })}
            </nav>
          </div>

          <div className="flex items-center gap-4">
            {/* API URL */}
            <div className="hidden sm:flex items-center gap-2">
              <Input
                className="w-48 text-xs"
                value={baseUrl}
                onChange={e => setBaseUrl(e.target.value)}
                placeholder="API URL"
              />
              <Button size="sm" variant="outline" onClick={handleTestConnection}>
                {apiReady ? "Connected" : "Connect"}
              </Button>
            </div>

            {/* Checklist items indicator */}
            <div className="flex items-center gap-1.5 text-xs text-muted-foreground/80 bg-muted/30 px-3 py-1.5 rounded-full">
              <div className={`h-2 w-2 rounded-full ${apiReady ? "bg-green-500" : "bg-red-500"}`} />
              <span className="font-medium mr-2">API</span>
              <div className={`h-2 w-2 rounded-full ${modelsReady ? "bg-green-500" : "bg-yellow-500"}`} />
              <span className="font-medium mr-2">Models</span>
              <div className={`h-2 w-2 rounded-full ${docsReady ? "bg-green-500" : "bg-yellow-500"}`} />
              <span className="font-medium">Knowledge</span>
            </div>

            {/* Theme Toggle */}
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setIsDarkMode(!isDarkMode)}
              className="text-foreground"
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
            <div className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "config" ? "hidden" : ""}`}>
              {/* Configuration Panel */}
              <div className="space-y-6 max-w-[1600px] mx-auto">
                <div>
                  <h2 className="text-lg font-semibold tracking-tight">System Configuration</h2>
                  <p className="text-sm text-muted-foreground">Manage LLM providers and retrieval settings.</p>
                </div>

                <div className="grid gap-6">
                  {/* Provider Carousel - Ollama, LM Studio, OpenRouter */}
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
                  />

                  {/* Advanced Provider Configuration */}
                  <ProviderConfig
                    config={config}
                    setConfig={setConfig}
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

                  {/* Advanced Engineering Presets */}
                  <div className="bg-card rounded-lg border border-border/60 p-6 shadow-sm">
                    <div className="flex items-center gap-2 mb-2">
                      <Settings className="h-5 w-5 text-primary" />
                      <h3 className="font-semibold">Advanced Engineering Presets</h3>
                    </div>
                    <p className="text-sm text-muted-foreground mb-4">
                      Apply tuned defaults for common workflows
                    </p>
                    <PresetSelector
                      value={activePreset}
                      onChange={async (newPreset: PresetLevel) => {
                        setActivePreset(newPreset);
                        try {
                          await fetchJson(`/config/presets/${newPreset}`, { method: "POST" });
                          toast({ title: `Applied ${newPreset} preset`, variant: "default" });
                        } catch (e) {
                          toast({ title: toMessage(e), variant: "error" });
                        }
                      }}
                      disabled={isSavingConfig}
                    />
                  </div>

                  {/* Retrieval Settings */}
                  <AdvancedRAGSettings
                    retrieval={config?.retrieval}
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
                  />

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
                                  <span>• {doc.chunk_count} chunks</span>
                                </div>
                              </div>
                              <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                <div className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${doc.metadata?.processing_status === 'ready' ? 'bg-green-500/10 text-green-600' :
                                  doc.metadata?.processing_status === 'error' ? 'bg-red-500/10 text-red-600' :
                                    'bg-yellow-500/10 text-yellow-600'
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
                onNewChat={handleNewChat}
                onCancel={handleCancelQuery}
                savedSessions={savedSessions}
                onLoadSession={handleLoadSession}
                onDeleteSession={handleDeleteSession}
                onClearHistory={handleClearHistory}
                scrollRef={chatEndRef}
                currentSessionId={currentSessionId}
                preset={activePreset}
                onPresetChange={async (preset) => {
                  setActivePreset(preset);
                  try {
                    await fetchJson(`/config/presets/${preset}`, { method: "POST" });
                    toast({ title: "Preset applied", description: `Switched to ${preset} mode`, variant: "success" });
                  } catch (e) {
                    console.error("Failed to apply preset:", e);
                  }
                }}
              />
            </div>

            <div className={`h-full animate-in fade-in slide-in-from-bottom-2 duration-300 ${activeTab !== "metrics" ? "hidden" : ""}`}>
              {/* Metrics & Traces */}
              <div className="space-y-6 max-w-5xl mx-auto">
                <EnterpriseStatusPanel baseUrl={baseUrl} />

                <MetricsDashboard traces={traces} />
                <TraceLog
                  isEvaluating={isEvaluating}
                  handleEvaluation={handleEvaluation}
                  evaluationSummary={evaluationSummary}
                  traces={traces}
                  formatNumber={formatNumber}
                />
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

import { useEffect, useMemo, useRef, useState } from "react";
import { BarChart3, FileText, MessageSquare, Moon, Settings, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useToast } from "@/components/ui/toast";

import { AdvancedRAGSettings } from "@/components/features/AdvancedRAGSettings";
import { ChatInterface } from "@/components/features/ChatInterface";
import { IngestPanel } from "@/components/features/IngestPanel";
import { MetricsDashboard } from "@/components/features/MetricsDashboard";
import { ProviderConfig } from "@/components/features/ProviderConfig";
import { TraceLog } from "@/components/features/TraceLog";

import "./index.css";
import type { AppConfig, CacheStats, DocumentOut, IngestResponse, LocalProviderInfo, ModelStatus, ProviderConfig as ProviderConfigType, ProviderProfile, QueryResponse, RetrievalDefaults, RoleSelection, TraceOut } from "@/types";

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
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [traces, setTraces] = useState<TraceOut[]>([]);
  const [cacheStats, setCacheStats] = useState<CacheStats | null>(null);
  const [question, setQuestion] = useState("What is JR AutoRAG?");
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [selectedDocumentIds, setSelectedDocumentIds] = useState<string[]>([]);
  const [ingestTitle, setIngestTitle] = useState("Getting Started");
  const [ingestText, setIngestText] = useState("Paste onboarding doc content here...");
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [activeStage, setActiveStage] = useState<string | null>(null);
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
  const [activeTab, setActiveTab] = useState<TabId>("config");
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

  // Toggle dark mode
  useEffect(() => {
    if (isDarkMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDarkMode]);

  const buildUrl = (path: string) => `${baseUrl.replace(/\/$/, "")}${path}`;

  const fetchJson = async <T,>(path: string, init?: RequestInit): Promise<T> => {
    const response = await fetch(buildUrl(path), init);
    if (!response.ok) {
      throw new Error(await response.text());
    }
    return response.json();
  };

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
      toast({ title: "API reachable", description: baseUrl, variant: "success" });
    } catch (error) {
      setStatus(`Health check failed: ${toMessage(error)}`);
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
        body: JSON.stringify({ title: ingestTitle, text: ingestText }),
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

  const handleAsk = async () => {
    setIsQuerying(true);
    setQueryResult(null);
    setActiveStage("planning");
    try {
      const useFilter = selectedDocumentIds.length > 0 && selectedDocumentIds.length < documents.length;
      const payload = useFilter ? { question, document_ids: selectedDocumentIds } : { question };
      const response = await fetch(buildUrl("/query/stream"), {
        method: "POST",
        headers,
        body: JSON.stringify(payload),
      });
      if (!response.ok || !response.body) {
        throw new Error(await response.text());
      }

      const reader = response.body.getReader();
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
          if (event.type === "token") {
            const text = event.data?.text ?? "";
            if (text) {
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
          if (event.type === "stage") {
            setActiveStage(event.data?.name ?? null);
          }

        if (event.type === "result") {
          receivedResult = true;
          setQueryResult(event.data);
          setStatus("Query succeeded");
          setActiveStage(null);
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
    setStatus(`Query failed: ${toMessage(error)}`);
    toast({ title: "Query failed", description: toMessage(error), variant: "error" });
  } finally {
    setIsQuerying(false);
    setActiveStage(null);
  }
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
    setIsClearingCache(true);
    try {
      await fetchJson("/monitoring/cache/clear", { method: "POST" });
      setStatus("Cache cleared");
      refreshAll();
      toast({ title: "Cache cleared", description: "Embedding and query caches reset", variant: "success" });
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
  const apiReady = status.toLowerCase().includes("api reachable");

  return (
    <div className="min-h-screen bg-background transition-colors duration-300">
      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-border/60 bg-background/95">
        <div className="mx-auto flex max-w-[1600px] items-center justify-between px-4 py-4">
          <div className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary text-lg font-bold text-primary-foreground shadow-sm">
                JR
              </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">AutoRAG</h1>
              <p className="text-xs text-muted-foreground">Admin Console</p>
            </div>
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
              <Button variant="outline" size="sm" onClick={handleTestConnection}>
                Test
              </Button>
            </div>

            {/* Dark Mode Toggle */}
            <button
              type="button"
              onClick={() => setIsDarkMode(!isDarkMode)}
              className="flex h-9 w-9 items-center justify-center rounded-md border border-border/70 bg-card text-lg transition-colors hover:bg-muted/40"
              aria-label="Toggle dark mode"
            >
              {isDarkMode ? <Moon className="h-4 w-4" /> : <Sun className="h-4 w-4" />}
            </button>
          </div>
        </div>

        {/* Status Bar */}
        {status && (
        <div className="border-t border-border/60 bg-muted/20 px-4 py-2">
            <p className="mx-auto max-w-[1600px] text-sm text-muted-foreground">
              <span className="inline-block h-2 w-2 rounded-full bg-primary mr-2" />
              {status}
            </p>
          </div>
        )}
      </header>

      {/* Tab Navigation */}
      <nav className="border-b border-border/60 bg-muted/20">
        <div className="mx-auto max-w-[1600px] px-4">
          <div className="flex gap-1 overflow-x-auto py-2">
            {tabs.map(tab => (
              <button
                key={tab.id}
                type="button"
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 rounded-md px-4 py-2 text-sm font-medium transition-colors ${activeTab === tab.id
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:bg-muted/40 hover:text-foreground"
                  }`}
              >
                <tab.icon className="h-4 w-4" />
                <span className="hidden sm:inline">{tab.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Setup Checklist */}
      {!checklistDismissed && (
        <div className="border-b border-border/60 bg-muted/10">
          <div className="mx-auto flex max-w-[1600px] flex-col gap-3 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
            <div className="flex flex-wrap items-center gap-3 text-sm text-foreground">
              <span className="font-semibold text-foreground">Setup Checklist</span>
              <span className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs ${docsReady ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
                <span className={`h-2 w-2 rounded-full ${docsReady ? "bg-emerald-500" : "bg-amber-500"}`} />
                {docsReady ? "Documents ready" : "Add documents"}
              </span>
              <span className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs ${modelsReady ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
                <span className={`h-2 w-2 rounded-full ${modelsReady ? "bg-emerald-500" : "bg-amber-500"}`} />
                {modelsReady ? "Models configured" : "Select models"}
              </span>
              <span className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs ${apiReady ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700"}`}>
                <span className={`h-2 w-2 rounded-full ${apiReady ? "bg-emerald-500" : "bg-amber-500"}`} />
                {apiReady ? "API reachable" : "Test API"}
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-muted px-2 py-1 text-xs text-muted-foreground">
                OCR: install tesseract + poppler for scanned PDFs
              </span>
            </div>
            <button
              type="button"
              onClick={() => setChecklistDismissed(true)}
              className="text-xs text-muted-foreground hover:text-foreground"
            >
              Dismiss
            </button>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="mx-auto max-w-[1600px] px-4 py-8">
        <div className="flex flex-col gap-8">
          {/* Config Tab */}
          {activeTab === "config" && (
            <>
              <ProviderConfig
                config={config}
                setConfig={setConfig}
                selectedProfile={selectedProfile}
                handleSelectProfile={handleSelectProfile}
                newProfileName={newProfileName}
                setNewProfileName={setNewProfileName}
                handleAddProfile={handleAddProfile}
                handleDiscoverModels={handleDiscoverModels}
                isSavingConfig={isSavingConfig}
                handleSaveConfig={handleSaveConfig}
                refreshAll={refreshAll}
                localProviders={localProviders}
                localProvidersStatus={localProvidersStatus}
                refreshLocalProviders={refreshLocalProviders}
                localSelections={localSelections}
                setLocalSelection={setLocalSelection}
                applyLocalProvider={applyLocalProvider}
                modelOptions={modelOptions}
              />
              <AdvancedRAGSettings
                retrieval={config?.retrieval}
                updateRetrieval={updateRetrieval}
                onSave={handleSaveConfig}
                isSaving={isSavingConfig}
                modelStatus={modelStatus}
                isCheckingModels={isCheckingModels}
                modelActionMessage={modelActionMessage}
                onRefreshModelStatus={() =>
                  refreshModelStatus(
                    config?.retrieval?.embedding_model,
                    config?.retrieval?.reranker_model,
                  )
                }
                onDownloadEmbedding={() => downloadModel("embedding")}
                onDownloadReranker={() => downloadModel("reranker")}
                onDeleteEmbedding={() => deleteModel("embedding")}
                onDeleteReranker={() => deleteModel("reranker")}
                isDownloadingEmbedding={isDownloadingEmbedding}
                isDownloadingReranker={isDownloadingReranker}
              />
            </>
          )}

          {/* Documents Tab */}
          {activeTab === "documents" && (
            <IngestPanel
              ingestTitle={ingestTitle}
              setIngestTitle={setIngestTitle}
              ingestText={ingestText}
              setIngestText={setIngestText}
              isIngesting={isIngesting}
              handleIngest={handleIngest}
              documents={documents}
              handleDeleteDocument={handleDeleteDocument}
              handleDeleteAllDocuments={handleDeleteAllDocuments}
              isUploadingFile={isUploadingFile}
              uploadFile={uploadFile}
              waitForDocumentReady={waitForDocumentReady}
              fileInputRef={fileInputRef}
              formatDateTime={formatDateTime}
            />
          )}

          {/* Query Tab */}
          {activeTab === "query" && (
            <ChatInterface
              question={question}
              setQuestion={setQuestion}
              isQuerying={isQuerying}
              handleAsk={handleAsk}
              queryResult={queryResult}
              documents={documents}
              selectedDocumentIds={selectedDocumentIds}
              setSelectedDocumentIds={setSelectedDocumentIds}
              providerConfig={config?.provider}
              activeStage={activeStage}
            />
          )}

          {/* Metrics Tab */}
          {activeTab === "metrics" && (
            <>
              <MetricsDashboard
                traces={traces}
                cacheStats={cacheStats ?? undefined}
                onClearCache={handleClearCache}
                isClearingCache={isClearingCache}
              />
              <TraceLog
                isEvaluating={isEvaluating}
                handleEvaluation={handleEvaluation}
                evaluationSummary={evaluationSummary}
                traces={traces}
                formatNumber={formatNumber}
              />
            </>
          )}
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-border bg-muted/30 py-4">
        <div className="mx-auto max-w-[1600px] px-4 text-center text-xs text-muted-foreground">
          JR AutoRAG • From Hello.World Consulting
        </div>
      </footer>
    </div>
  );
}

export default App;

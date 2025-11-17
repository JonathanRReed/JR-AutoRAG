import { useEffect, useMemo, useRef, useState, type ChangeEvent, type DragEvent } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";

import "./index.css";

type ProviderConfig = {
  name: string;
  base_url: string;
  planner_model?: string;
  gatherer_model?: string;
  generator_model?: string;
  api_key?: string;
};

type ProviderProfile = {
  name: string;
  provider: ProviderConfig;
};

type RetrievalDefaults = {
  hybrid: boolean;
  dense_k: number;
  sparse_k: number;
  rerank_pool: number;
  top_n: number;
  compression: boolean;
  target_tokens: number;
  raptor: string;
  graph: boolean;
  coverage_target: number;
  max_context_tokens: number;
};

type AppConfig = {
  profile: string;
  provider?: ProviderConfig;
  provider_profiles: ProviderProfile[];
  retrieval: RetrievalDefaults;
};

type DocumentOut = {
  id: string;
  title: string;
  text: string;
  metadata: Record<string, string>;
};

type ChunkOut = {
  id: string;
  title: string;
  snippet: string;
  score: number;
};

type QueryResponse = {
  answer: string;
  chunks: ChunkOut[];
  trace_id: string;
  metrics: Record<string, number>;
};

type TraceOut = {
  id: string;
  prompt: string;
  answer: string;
  metrics: Record<string, number>;
};

type ProviderKind = "ollama" | "lmstudio" | "openai";

type LocalProviderInfo = {
  kind: ProviderKind;
  name: string;
  base_url: string;
  models: string[];
  running: string[];
  version?: string;
  status?: string;
  error_message?: string;
};

type RoleSelection = {
  planner: string;
  gatherer: string;
  generator: string;
};

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

export function App() {
  const [baseUrl, setBaseUrl] = useState(defaultBaseUrl);
  const [status, setStatus] = useState("");
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [traces, setTraces] = useState<TraceOut[]>([]);
  const [question, setQuestion] = useState("What is JR AutoRAG?");
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);
  const [ingestTitle, setIngestTitle] = useState("Getting Started");
  const [ingestText, setIngestText] = useState("Paste onboarding doc content here...");
  const [isSavingConfig, setIsSavingConfig] = useState(false);
  const [isIngesting, setIsIngesting] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const [evaluationSummary, setEvaluationSummary] = useState("");
  const [selectedProfile, setSelectedProfile] = useState("Default");
  const [newProfileName, setNewProfileName] = useState("Default");
  const [modelOptions, setModelOptions] = useState<string[]>([]);
  const [localProviders, setLocalProviders] = useState<LocalProviderInfo[]>([]);
  const [localProvidersStatus, setLocalProvidersStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const [localSelections, setLocalSelections] = useState<Record<string, RoleSelection>>({});
  const [isUploadingFile, setIsUploadingFile] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const headers = useMemo(() => ({ "Content-Type": "application/json" }), []);

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

  const handleDeleteDocument = async (id: string, title: string) => {
    const confirmed = window.confirm(`Delete “${title}”? This cannot be undone.`);
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

  const refreshAll = async () => {
    try {
      const [cfg, docs, traceList] = await Promise.all([
        fetchJson<AppConfig>("/config"),
        fetchJson<DocumentOut[]>("/documents"),
        fetchJson<TraceOut[]>("/monitoring/traces"),
      ]);
      setConfig(cfg);
      setSelectedProfile(cfg.profile);
      setNewProfileName(cfg.profile);
      setDocuments(docs);
      setTraces(traceList);
      setStatus("API data loaded");
    } catch (error) {
      setStatus(`Failed to load data: ${toMessage(error)}`);
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

  const handleTestConnection = async () => {
    setStatus("Testing connection...");
    try {
      await fetchJson("/healthz");
      setStatus("API reachable");
    } catch (error) {
      setStatus(`Health check failed: ${toMessage(error)}`);
    }
  };

  const updateProvider = (field: keyof ProviderConfig, value: string) => {
    setConfig(cfg =>
      cfg
        ? {
            ...cfg,
            provider: {
              name: cfg.provider?.name ?? "",
              base_url: cfg.provider?.base_url ?? "",
              ...cfg.provider,
              [field]: value,
            },
          }
        : cfg,
    );
  };

  const updateRetrieval = (field: keyof RetrievalDefaults, value: string | number | boolean) => {
    setConfig(cfg => (cfg ? { ...cfg, retrieval: { ...cfg.retrieval, [field]: value } } : cfg));
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
    } catch (error) {
      console.error("Error saving configuration:", error);
      setStatus(`Save failed: ${toMessage(error)}`);
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
    } catch (error) {
      console.error("Error discovering models:", error);
      setStatus(`Model discovery failed: ${toMessage(error)}`);
    }
  };

  const handleIngest = async () => {
    setIsIngesting(true);
    try {
      await fetchJson("/documents/text", {
        method: "POST",
        headers,
        body: JSON.stringify({ title: ingestTitle, text: ingestText }),
      });
      setStatus("Document ingested");
      setIngestText("");
      refreshAll();
    } catch (error) {
      setStatus(`Ingest failed: ${toMessage(error)}`);
    } finally {
      setIsIngesting(false);
    }
  };

  const handleAsk = async () => {
    setIsQuerying(true);
    try {
      const result = await fetchJson<QueryResponse>("/query", {
        method: "POST",
        headers,
        body: JSON.stringify({ question }),
      });
      setQueryResult(result);
      setStatus("Query succeeded");
      refreshAll();
    } catch (error) {
      setStatus(`Query failed: ${toMessage(error)}`);
    } finally {
      setIsQuerying(false);
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
    } catch (error) {
      setStatus(`Evaluation failed: ${toMessage(error)}`);
    } finally {
      setIsEvaluating(false);
    }
  };

  const profileOptions = useMemo(() => {
    const names = new Set<string>([selectedProfile]);
    (config?.provider_profiles ?? []).forEach(profile => names.add(profile.name));
    return Array.from(names);
  }, [config, selectedProfile]);

  const configProvider = config?.provider;
  const retrieval = config?.retrieval;

  const handleNumber = (event: ChangeEvent<HTMLInputElement>, field: keyof RetrievalDefaults) => {
    updateRetrieval(field, Number(event.target.value));
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
    const selection = localSelections[provider.base_url] ?? {
      planner: provider.running[0] ?? provider.models[0] ?? "",
      gatherer: provider.running[0] ?? provider.models[0] ?? "",
      generator: provider.running[0] ?? provider.models[0] ?? "",
    };
    if (!config) {
      setStatus("Configuration not loaded yet. Please wait and try again.");
      return;
    }
    if (!provider.models.length) {
      setStatus(`No models available for ${provider.name}. Install or run one first.`);
      return;
    }
    const nextConfig: AppConfig = {
      ...config,
      provider: {
        name: provider.name,
        base_url: provider.base_url,
        planner_model: selection.planner || provider.models[0],
        gatherer_model: selection.gatherer || provider.models[0],
        generator_model: selection.generator || provider.models[0],
      },
    };
    setConfig(nextConfig);
    try {
      await persistConfig(nextConfig, `Saved ${provider.name} provider settings`);
    } catch {
      // status already updated inside persistConfig
    }
  };

  const uploadFile = async (file: File) => {
    setIsUploadingFile(true);
    setStatus(`Uploading ${file.name}...`);
    try {
      const formData = new FormData();
      const title = file.name.replace(/\.[^.]+$/, "");
      formData.append("title", title || file.name);
      formData.append("file", file);
      await fetch(buildUrl("/documents/upload"), {
        method: "POST",
        body: formData,
      });
      setStatus(`Uploaded ${file.name}`);
      setIngestTitle(title || file.name);
      setIngestText("");
      refreshAll();
    } catch (error) {
      setStatus(`Upload failed: ${toMessage(error)}`);
    } finally {
      setIsUploadingFile(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleFileInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      void uploadFile(file);
    }
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (file) {
      void uploadFile(file);
    }
  };

  const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const renderProviderModels = (provider: LocalProviderInfo, field: keyof RoleSelection, label: string) => {
    const selection = localSelections[provider.base_url];
    return (
      <div className="space-y-1">
        <Label className="text-xs text-muted-foreground">{label}</Label>
        <Select
          value={selection?.[field] ?? ""}
          onValueChange={value => setLocalSelection(provider.base_url, field, value)}
          disabled={!provider.models.length}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select model" />
          </SelectTrigger>
          <SelectContent>
            {provider.models.map(model => (
              <SelectItem key={model} value={model}>
                {model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-muted/20 px-4 py-10">
      <div className="mx-auto flex max-w-6xl flex-col gap-8">
        <header className="flex flex-col gap-4 rounded-2xl bg-white/90 p-6 shadow-sm sm:flex-row sm:items-center sm:justify-between">
          <div>
            <p className="text-sm uppercase tracking-wide text-muted-foreground">JR AutoRAG</p>
            <h1 className="text-3xl font-semibold">Admin Console</h1>
            <p className="text-muted-foreground">Configure providers, ingest docs, run queries, monitor traces.</p>
          </div>
          <div className="flex flex-col gap-2 sm:w-80">
            <Label className="text-xs font-semibold text-muted-foreground">API Base URL</Label>
            <div className="flex gap-2">
              <Input value={baseUrl} onChange={e => setBaseUrl(e.target.value)} />
              <Button variant="secondary" onClick={handleTestConnection}>
                Test
              </Button>
            </div>
            <span className="text-sm text-blue-600">{status}</span>
          </div>
        </header>

        <Card>
          <CardHeader>
            <CardTitle>Quick Setup</CardTitle>
            <CardDescription>Define provider defaults and save reusable profiles.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid gap-4 sm:grid-cols-[minmax(220px,260px)_1fr]">
              <div className="space-y-4">
                <Label>Active Profile</Label>
                <Select value={selectedProfile} onValueChange={handleSelectProfile}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose profile" />
                  </SelectTrigger>
                  <SelectContent>
                    {profileOptions.map(name => (
                      <SelectItem key={name} value={name}>
                        {name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <div className="space-y-2">
                  <Label htmlFor="newProfile">Save as</Label>
                  <Input id="newProfile" value={newProfileName} onChange={e => setNewProfileName(e.target.value)} />
                  <div className="flex gap-2">
                    <Button variant="secondary" onClick={handleAddProfile}>
                      Save Profile
                    </Button>
                    <Button variant="outline" onClick={handleDiscoverModels}>
                      Discover Models
                    </Button>
                  </div>
                </div>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                <div className="space-y-2">
                  <Label htmlFor="providerName">Provider Name</Label>
                  <Input
                    id="providerName"
                    value={configProvider?.name ?? ""}
                    onChange={e => updateProvider("name", e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="providerUrl">Provider Base URL</Label>
                  <Input
                    id="providerUrl"
                    value={configProvider?.base_url ?? ""}
                    onChange={e => updateProvider("base_url", e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="plannerModel">Planner Model</Label>
                  <Input
                    id="plannerModel"
                    list="modelOptions"
                    value={configProvider?.planner_model ?? ""}
                    onChange={e => updateProvider("planner_model", e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="generatorModel">Generator Model</Label>
                  <Input
                    id="generatorModel"
                    list="modelOptions"
                    value={configProvider?.generator_model ?? ""}
                    onChange={e => updateProvider("generator_model", e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="gathererModel">Gatherer Model</Label>
                  <Input
                    id="gathererModel"
                    list="modelOptions"
                    value={configProvider?.gatherer_model ?? ""}
                    onChange={e => updateProvider("gatherer_model", e.target.value)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="apiKey">API Key</Label>
                  <Input
                    id="apiKey"
                    type="password"
                    value={configProvider?.api_key ?? ""}
                    onChange={e => updateProvider("api_key", e.target.value)}
                  />
                </div>
              </div>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="denseK">Dense k</Label>
                <Input
                  id="denseK"
                  type="number"
                  value={retrieval?.dense_k ?? 0}
                  onChange={event => handleNumber(event, "dense_k")}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="sparseK">Sparse k</Label>
                <Input
                  id="sparseK"
                  type="number"
                  value={retrieval?.sparse_k ?? 0}
                  onChange={event => handleNumber(event, "sparse_k")}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="rerankPool">Rerank pool</Label>
                <Input
                  id="rerankPool"
                  type="number"
                  value={retrieval?.rerank_pool ?? 0}
                  onChange={event => handleNumber(event, "rerank_pool")}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="topN">Final top N</Label>
                <Input
                  id="topN"
                  type="number"
                  value={retrieval?.top_n ?? 0}
                  onChange={event => handleNumber(event, "top_n")}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="coverage">Coverage target</Label>
                <Input
                  id="coverage"
                  type="number"
                  step="0.05"
                  value={retrieval?.coverage_target ?? 0}
                  onChange={event => handleNumber(event, "coverage_target")}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="context">Max context tokens</Label>
                <Input
                  id="context"
                  type="number"
                  value={retrieval?.max_context_tokens ?? 0}
                  onChange={event => handleNumber(event, "max_context_tokens")}
                />
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={retrieval?.hybrid ?? false}
                  onChange={event => updateRetrieval("hybrid", event.target.checked)}
                />
                Hybrid search
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={retrieval?.compression ?? false}
                  onChange={event => updateRetrieval("compression", event.target.checked)}
                />
                Compression
              </label>
              <label className="flex items-center gap-2 text-sm">
                <input
                  type="checkbox"
                  className="h-4 w-4"
                  checked={retrieval?.graph ?? false}
                  onChange={event => updateRetrieval("graph", event.target.checked)}
                />
                Graph mode
              </label>
            </div>

            <div className="flex flex-wrap gap-3">
              <Button onClick={handleSaveConfig} disabled={isSavingConfig}>
                {isSavingConfig ? "Saving..." : "Save Configuration"}
              </Button>
              <Button variant="outline" onClick={refreshAll}>
                Refresh
              </Button>
            </div>

            {(config?.provider_profiles?.length ?? 0) > 0 && (
              <div>
                <p className="text-sm font-semibold">Profiles</p>
                <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                  {config?.provider_profiles?.map(profile => (
                    <li key={profile.name}>
                      <span className="font-medium text-foreground">{profile.name}</span> · {profile.provider.name} @
                      {" "}
                      {profile.provider.base_url}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Detected Local Providers</CardTitle>
            <CardDescription>
              Auto-detect running Ollama or LM Studio runtimes and apply their models with one click.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between text-sm text-muted-foreground">
              <span>
                {localProvidersStatus === "loading" && "Scanning for runtimes..."}
                {localProvidersStatus === "ready" && `Found ${localProviders.length} provider${localProviders.length === 1 ? "" : "s"}.`}
                {localProvidersStatus === "error" && "Unable to reach local runtimes."}
                {localProvidersStatus === "idle" && "Waiting to scan..."}
              </span>
              <Button variant="outline" size="sm" onClick={refreshLocalProviders} disabled={localProvidersStatus === "loading"}>
                {localProvidersStatus === "loading" ? "Scanning..." : "Rescan"}
              </Button>
            </div>
            {config?.provider && (
              <div className="rounded-lg border border-border bg-muted/40 px-4 py-3 text-sm text-muted-foreground">
                Active provider: <span className="font-semibold text-foreground">{config.provider.name}</span> @
                <span className="text-foreground"> {config.provider.base_url}</span>
                {config.provider.generator_model && ` · Generator: ${config.provider.generator_model}`}
              </div>
            )}
            {localProviders.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                Start Ollama (`ollama serve`) or LM Studio and click Rescan to prefill provider settings.
              </p>
            ) : (
              <div className="space-y-4">
                {localProviders.map(provider => {
                  const running = provider.running.filter(Boolean);
                  const models = provider.models;
                  const badgeLabel = provider.kind === "ollama" ? "Ollama" : provider.kind === "lmstudio" ? "LM Studio" : "OpenAI";
                  return (
                    <div key={provider.base_url} className="rounded-lg border border-border p-4 space-y-4">
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
                        <div>
                          <p className="font-semibold text-foreground">{provider.name}</p>
                          <p className="text-xs text-muted-foreground">{provider.base_url}</p>
                        </div>
                        <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-wide">
                          <span className="rounded-full bg-muted px-2 py-1 text-muted-foreground">{badgeLabel}</span>
                          {provider.version && <span className="text-muted-foreground">v{provider.version}</span>}
                          {provider.status === "error" && (
                            <span className="rounded-full bg-destructive/10 px-2 py-1 text-destructive">
                              Warning
                            </span>
                          )}
                        </div>
                      </div>
                      {provider.status === "error" && provider.error_message && (
                        <p className="text-sm text-destructive">
                          {provider.error_message}
                        </p>
                      )}
                      <div className="text-sm text-muted-foreground space-y-1">
                        {running.length ? (
                          <p>
                            Running: <span className="text-foreground">{running.join(", ")}</span>
                          </p>
                        ) : (
                          <p>No models currently loaded.</p>
                        )}
                        <p>{models.length ? `${models.length} model${models.length === 1 ? "" : "s"} available.` : "No models found."}</p>
                      </div>
                      <div className="grid gap-3 md:grid-cols-3">
                        {renderProviderModels(provider, "planner", "Planner model")}
                        {renderProviderModels(provider, "gatherer", "Gatherer model")}
                        {renderProviderModels(provider, "generator", "Generator model")}
                      </div>
                      <Button
                        onClick={() => void applyLocalProvider(provider)}
                        disabled={!models.length || isSavingConfig}
                        variant="secondary"
                      >
                        Use {provider.name}
                      </Button>
                    </div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Ingest Documents</CardTitle>
            <CardDescription>Load PDFs, wikis or notes to make them queryable.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div
              className="rounded-lg border border-dashed border-border p-6 text-center transition hover:border-foreground"
              onDrop={handleDrop}
              onDragOver={handleDragOver}
            >
              <p className="font-medium">Drop a PDF, DOCX, or TXT here</p>
              <p className="text-sm text-muted-foreground">Files are chunked locally before indexing.</p>
              <div className="mt-3 flex flex-col items-center gap-2">
                <input
                  type="file"
                  accept=".pdf,.doc,.docx,.txt"
                  ref={fileInputRef}
                  onChange={handleFileInputChange}
                  className="sr-only"
                  id="document-upload-input"
                />
                <Button variant="outline" onClick={() => fileInputRef.current?.click()} disabled={isUploadingFile}>
                  {isUploadingFile ? "Uploading..." : "Select File"}
                </Button>
                <p className="text-xs text-muted-foreground">Supports PDFs, Word docs, and plain text up to 25 MB.</p>
              </div>
            </div>
            <div className="flex flex-col gap-4 lg:flex-row">
              <div className="basis-full space-y-2 lg:basis-1/2">
                <Label htmlFor="ingestTitle">Title</Label>
                <Input id="ingestTitle" value={ingestTitle} onChange={e => setIngestTitle(e.target.value)} />
                <Button disabled={isIngesting} onClick={handleIngest}>
                  {isIngesting ? "Ingesting..." : "Ingest Text"}
                </Button>
              </div>
              <div className="basis-full space-y-2 lg:basis-1/2">
                <Label htmlFor="ingestText">Content</Label>
                <Textarea
                  id="ingestText"
                  rows={6}
                  value={ingestText}
                  onChange={e => setIngestText(e.target.value)}
                  placeholder="Paste doc text or summary"
                />
              </div>
            </div>
            <div>
              <p className="text-sm font-semibold">Documents ({documents.length})</p>
              {documents.length === 0 ? (
                <p className="mt-2 text-sm text-muted-foreground">No documents ingested yet.</p>
              ) : (
                <div className="mt-3 grid gap-3 md:grid-cols-2">
                  {documents.map(doc => {
                    const uploadedAt = formatDateTime(doc.metadata?.uploaded_at);
                    const filesize = doc.metadata?.filesize
                      ? `${(Number(doc.metadata.filesize) / (1024 * 1024)).toFixed(2)} MB`
                      : undefined;
                    const filename = doc.metadata?.original_filename || doc.metadata?.filename;
                    return (
                      <div key={doc.id} className="rounded-lg border border-border p-3 text-sm">
                        <div className="flex items-start justify-between gap-2">
                          <div>
                            <p className="font-semibold text-foreground">{doc.title || filename || "Untitled"}</p>
                            {filename && <p className="text-xs text-muted-foreground">{filename}</p>}
                          </div>
                          <Button
                            variant="ghost"
                            size="sm"
                            className="text-destructive hover:bg-destructive/10"
                            onClick={() => void handleDeleteDocument(doc.id, doc.title || filename || "document")}
                          >
                            Remove
                          </Button>
                        </div>
                        <p className="mt-2 text-muted-foreground">{doc.text.slice(0, 140) || "(Empty document)"}…</p>
                        <div className="mt-3 flex flex-wrap gap-3 text-xs text-muted-foreground">
                          <span>Uploaded {uploadedAt}</span>
                          {filesize && <span>Size {filesize}</span>}
                          {doc.metadata?.content_type && <span>{doc.metadata.content_type}</span>}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Ask a Question</CardTitle>
            <CardDescription>Send a query through the full RAG stack.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col gap-2 sm:flex-row">
              <Input value={question} onChange={e => setQuestion(e.target.value)} />
              <Button disabled={isQuerying} onClick={handleAsk}>
                {isQuerying ? "Querying..." : "Run Query"}
              </Button>
            </div>
            {queryResult && (
              <div className="space-y-4 rounded-lg border border-border p-4">
                <div>
                  <p className="text-sm font-semibold text-muted-foreground">Answer</p>
                  <p className="text-base text-foreground">{queryResult.answer}</p>
                </div>
                <div>
                  <p className="text-sm font-semibold text-muted-foreground">Evidence ({queryResult.chunks.length})</p>
                  <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                    {queryResult.chunks.map(chunk => (
                      <li key={chunk.id}>
                        <span className="font-medium text-foreground">{chunk.title}</span> · score {chunk.score.toFixed(3)} ·
                        {" "}
                        {chunk.snippet.slice(0, 120)}
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Evaluation & Monitoring</CardTitle>
            <CardDescription>Spot regressions and inspect traces.</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
              <Button disabled={isEvaluating} onClick={handleEvaluation}>
                {isEvaluating ? "Running..." : "Run Quick Evaluation"}
              </Button>
              {evaluationSummary && <p className="text-sm text-muted-foreground">{evaluationSummary}</p>}
            </div>
            <div>
              <p className="text-sm font-semibold">Traces ({traces.length})</p>
              <div className="mt-2 max-h-64 space-y-3 overflow-y-auto rounded-lg border border-border p-3 text-sm text-muted-foreground">
                {traces.map(trace => (
                  <div key={trace.id} className="border-b border-border/60 pb-2 last:border-b-0 last:pb-0">
                    <p className="font-medium text-foreground">{trace.prompt}</p>
                    <p>{trace.answer.slice(0, 140)}...</p>
                    <small className="text-xs text-muted-foreground">
                      coverage {formatNumber(trace.metrics.coverage)} · tokens {formatNumber(trace.metrics.tokens)}
                    </small>
                  </div>
                ))}
                {traces.length === 0 && <p>No traces yet. Run a query to capture one.</p>}
              </div>
            </div>
          </CardContent>
        </Card>

        <datalist id="modelOptions">
          {modelOptions.map(model => (
            <option key={model} value={model} />
          ))}
        </datalist>
      </div>
    </div>
  );
}

export default App;

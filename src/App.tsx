import { useEffect, useMemo, useRef, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

import { ChatInterface } from "@/components/features/ChatInterface";
import { IngestPanel } from "@/components/features/IngestPanel";
import { ProviderConfig } from "@/components/features/ProviderConfig";
import { TraceLog } from "@/components/features/TraceLog";

import "./index.css";
import type { AppConfig, DocumentOut, LocalProviderInfo, ProviderConfig as ProviderConfigType, ProviderProfile, QueryResponse, RetrievalDefaults, RoleSelection, TraceOut } from "@/types";

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
      const resp = await fetch(buildUrl("/documents/upload"), {
        method: "POST",
        body: formData,
      });
      if (!resp.ok) {
        const err = await resp.json().catch(() => ({ detail: resp.statusText }));
        throw new Error(err.detail || "Upload failed");
      }
      const result = await resp.json();
      setStatus(`Uploaded ${file.name} (${result.chunk_count} chunks)`);
      setIngestTitle("");
      setIngestText("");
      refreshAll();
    } catch (error) {
      setStatus(`Upload failed: ${toMessage(error)}`);
      throw error; // Re-throw so IngestPanel can mark as error
    } finally {
      setIsUploadingFile(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
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

        <IngestPanel
          ingestTitle={ingestTitle}
          setIngestTitle={setIngestTitle}
          ingestText={ingestText}
          setIngestText={setIngestText}
          isIngesting={isIngesting}
          handleIngest={handleIngest}
          documents={documents}
          handleDeleteDocument={handleDeleteDocument}
          isUploadingFile={isUploadingFile}
          uploadFile={uploadFile}
          fileInputRef={fileInputRef}
          formatDateTime={formatDateTime}
        />

        <ChatInterface
          question={question}
          setQuestion={setQuestion}
          isQuerying={isQuerying}
          handleAsk={handleAsk}
          queryResult={queryResult}
        />

        <TraceLog
          isEvaluating={isEvaluating}
          handleEvaluation={handleEvaluation}
          evaluationSummary={evaluationSummary}
          traces={traces}
          formatNumber={formatNumber}
        />
      </div>
    </div>
  );
}

export default App;

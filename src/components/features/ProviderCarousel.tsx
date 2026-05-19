import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CheckCircle2, Loader2, ChevronLeft, ChevronRight, Server, Cloud, Shield, ExternalLink } from "lucide-react";
import type { AppConfig, LocalProviderInfo, RoleSelection, OpenRouterStatus, OpenRouterModel, RAGFuzzStatus } from "@/types";

type ProviderTab = "ollama" | "ollama_cloud" | "lmstudio" | "openrouter";

interface ProviderCarouselProps {
  config: AppConfig | null;
  setConfig: React.Dispatch<React.SetStateAction<AppConfig | null>>;
  isSavingConfig: boolean;
  handleSaveConfig: () => void;
  persistConfig: (cfg: AppConfig, message?: string) => Promise<void>;
  localProviders: LocalProviderInfo[];
  localProvidersStatus: "idle" | "loading" | "ready" | "error";
  refreshLocalProviders: () => void;
  localSelections: Record<string, RoleSelection>;
  setLocalSelection: (baseUrl: string, field: keyof RoleSelection, value: string) => void;
  applyLocalProvider: (provider: LocalProviderInfo) => void;
  apiBaseUrl?: string;
  apiKey?: string;
  isConnected?: boolean;
}

export function ProviderCarousel({
  config,
  setConfig,
  isSavingConfig,
  handleSaveConfig,
  persistConfig,
  localProviders,
  localProvidersStatus,
  refreshLocalProviders,
  localSelections,
  setLocalSelection,
  applyLocalProvider,
  apiBaseUrl = "",
  apiKey = "",
  isConnected = false,
}: ProviderCarouselProps) {
  const [activeTab, setActiveTab] = useState<ProviderTab>("ollama");
  
  // OpenRouter state
  const [openRouterStatus, setOpenRouterStatus] = useState<OpenRouterStatus | null>(null);
  const [openRouterModels, setOpenRouterModels] = useState<OpenRouterModel[]>([]);
  const [openRouterLoading, setOpenRouterLoading] = useState(false);
  const [openRouterApiKey, setOpenRouterApiKey] = useState("");
  const [openRouterFilter, setOpenRouterFilter] = useState("");
  const [openRouterSelection, setOpenRouterSelection] = useState<RoleSelection>({
    planner: "",
    gatherer: "",
    generator: "",
  });

  // RAGFuzz state
  const [ragfuzzStatus, setRagfuzzStatus] = useState<RAGFuzzStatus | null>(null);
  const [ollamaFilter, setOllamaFilter] = useState("");
  const [lmstudioFilter, setLmstudioFilter] = useState("");
  
  // Ollama Cloud state
  const [ollamaCloudApiKey, setOllamaCloudApiKey] = useState("");
  const [ollamaCloudModels, setOllamaCloudModels] = useState<string[]>([]);
  const [ollamaCloudLoading, setOllamaCloudLoading] = useState(false);
  const [ollamaCloudFilter, setOllamaCloudFilter] = useState("");
  const [ollamaCloudSelection, setOllamaCloudSelection] = useState<RoleSelection>({
    planner: "",
    gatherer: "",
    generator: "",
  });

  const filterModels = (models: string[], term: string) => {
    if (!term.trim()) return models;
    const q = term.toLowerCase();
    return models.filter(m => m.toLowerCase().includes(q));
  };

  const filterOpenRouterModels = (models: OpenRouterModel[], term: string) => {
    if (!term.trim()) return models;
    const q = term.toLowerCase();
    return models.filter(m => (m.name || m.id).toLowerCase().includes(q));
  };

  const tabs: { id: ProviderTab; label: string; icon: React.ReactNode; color: string }[] = [
    { id: "ollama", label: "Ollama", icon: <Server className="h-4 w-4" />, color: "text-primary" },
    { id: "ollama_cloud", label: "Ollama Cloud", icon: <Cloud className="h-4 w-4" />, color: "text-primary" },
    { id: "lmstudio", label: "LM Studio", icon: <Server className="h-4 w-4" />, color: "text-muted-foreground" },
    { id: "openrouter", label: "OpenRouter", icon: <Cloud className="h-4 w-4" />, color: "text-muted-foreground" },
  ];

  const currentIndex = tabs.findIndex(t => t.id === activeTab);

  const goNext = () => {
    const next = (currentIndex + 1) % tabs.length;
    const nextTab = tabs[next];
    if (nextTab) setActiveTab(nextTab.id);
  };

  const goPrev = () => {
    const prev = (currentIndex - 1 + tabs.length) % tabs.length;
    const prevTab = tabs[prev];
    if (prevTab) setActiveTab(prevTab.id);
  };

  useEffect(() => {
    refreshLocalProviders();
    if (apiBaseUrl && isConnected) {
      fetchOpenRouterStatus();
      fetchRagfuzzStatus();
    }
  }, [apiBaseUrl, isConnected]);

  const fetchOpenRouterStatus = async () => {
    const authHeaders: Record<string, string> = {};
    const trimmedApiKey = apiKey.trim();
    if (trimmedApiKey) {
      authHeaders["X-API-Key"] = trimmedApiKey;
    }
    if (openRouterApiKey) {
      authHeaders["x-openrouter-key"] = openRouterApiKey;
      authHeaders.Authorization = `Bearer ${openRouterApiKey}`;
    }
    try {
      const res = await fetch(`${apiBaseUrl}/providers/openrouter/status`, {
        headers: Object.keys(authHeaders).length ? authHeaders : undefined,
      });
      if (res.ok) {
        const data = await res.json();
        setOpenRouterStatus(data);
        if (data.available) {
          fetchOpenRouterModels();
        }
      }
      // Silently ignore 429 rate limit errors - will retry on next render
    } catch {
      // Network errors are expected during API startup
    }
  };

  const fetchOpenRouterModels = async () => {
    setOpenRouterLoading(true);
    const authHeaders: Record<string, string> = {};
    const trimmedApiKey = apiKey.trim();
    if (trimmedApiKey) {
      authHeaders["X-API-Key"] = trimmedApiKey;
    }
    if (openRouterApiKey) {
      authHeaders["x-openrouter-key"] = openRouterApiKey;
      authHeaders.Authorization = `Bearer ${openRouterApiKey}`;
    }
    try {
      const res = await fetch(`${apiBaseUrl}/providers/openrouter/models`, {
        headers: Object.keys(authHeaders).length ? authHeaders : undefined,
      });
      if (res.ok) {
        const data = await res.json();
        setOpenRouterModels(data);
      }
    } catch (err) {
      console.error("Failed to fetch OpenRouter models", err);
    } finally {
      setOpenRouterLoading(false);
    }
  };

  const fetchRagfuzzStatus = async () => {
    const authHeaders: Record<string, string> = {};
    const trimmedApiKey = apiKey.trim();
    if (trimmedApiKey) {
      authHeaders["X-API-Key"] = trimmedApiKey;
    }
    try {
      const res = await fetch(`${apiBaseUrl}/rag/audit/health`, {
        headers: Object.keys(authHeaders).length ? authHeaders : undefined,
      });
      if (res.ok) {
        const data = await res.json();
        setRagfuzzStatus(data);
      }
    } catch {
      setRagfuzzStatus(null);
    }
  };

  const applyOpenRouter = async () => {
    if (!config) return;
    const model = openRouterSelection.planner || openRouterStatus?.default_model || "openai/gpt-4o-mini";
    const selected = openRouterSelection.planner || openRouterSelection.generator || openRouterSelection.gatherer;
    const nextConfig: AppConfig = {
      ...config,
      provider: {
        name: "OpenRouter",
        base_url: "https://openrouter.ai/api/v1",
        planner_model: openRouterSelection.planner || selected || model,
        generator_model: openRouterSelection.generator || selected || model,
        gatherer_model: openRouterSelection.gatherer || selected || model,
        api_key: openRouterApiKey || config.provider?.api_key || "",
      },
    };
    setConfig(nextConfig);
    await persistConfig(nextConfig, `Applied OpenRouter (${nextConfig.provider?.planner_model})`);
  };

  const [ollamaCloudError, setOllamaCloudError] = useState<string | null>(null);
  
  const fetchOllamaCloudModels = async () => {
    if (!ollamaCloudApiKey) {
      setOllamaCloudError("Please enter your Ollama API key first");
      return;
    }
    setOllamaCloudLoading(true);
    setOllamaCloudError(null);
    try {
      const authHeaders: Record<string, string> = {
        "x-ollama-key": ollamaCloudApiKey,
        Authorization: `Bearer ${ollamaCloudApiKey}`,
      };
      const trimmedApiKey = apiKey.trim();
      if (trimmedApiKey) {
        authHeaders["X-API-Key"] = trimmedApiKey;
      }
      const res = await fetch(`${apiBaseUrl}/providers/ollama-cloud/models`, {
        headers: authHeaders,
      });
      if (res.ok) {
        const data = await res.json();
        setOllamaCloudModels(data);
        if (data.length === 0) {
          setOllamaCloudError("No models available. Check your API key.");
        }
      } else {
        const errorData = await res.json().catch(() => ({ detail: res.statusText }));
        setOllamaCloudError(errorData.detail || `Error: ${res.status}`);
      }
    } catch (err) {
      console.error("Failed to fetch Ollama Cloud models", err);
      setOllamaCloudError("Failed to connect to Ollama Cloud. Check your network.");
    } finally {
      setOllamaCloudLoading(false);
    }
  };

  const applyOllamaCloud = async () => {
    if (!config) return;
    const selected = ollamaCloudSelection.planner || ollamaCloudSelection.generator || ollamaCloudSelection.gatherer || "llama3";
    const nextConfig: AppConfig = {
      ...config,
      provider: {
        name: "Ollama Cloud",
        base_url: "https://ollama.com",
        planner_model: ollamaCloudSelection.planner || selected,
        generator_model: ollamaCloudSelection.generator || selected,
        gatherer_model: ollamaCloudSelection.gatherer || selected,
        api_key: ollamaCloudApiKey || config.provider?.api_key || "",
      },
    };
    setConfig(nextConfig);
    await persistConfig(nextConfig, `Applied Ollama Cloud (${nextConfig.provider?.planner_model})`);
  };

  // Get provider by type
  const ollamaProvider = localProviders.find(p => p.kind === "ollama");
  const lmstudioProvider = localProviders.find(p => p.kind === "lmstudio");

  const renderModelSelect = (
    provider: LocalProviderInfo | undefined,
    field: keyof RoleSelection | "all",
    label: string,
    filterTerm: string
  ) => {
    if (!provider) return null;
    const selection = localSelections[provider.base_url];
    const value = field === "all" ? (selection?.planner || "") : (selection?.[field] ?? "");
    const models = filterModels(provider.models, filterTerm);

    return (
      <div className="space-y-1.5">
        <Label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">{label}</Label>
        <Select
          value={value}
          onValueChange={(val: string) => {
            if (field === "all") {
              setLocalSelection(provider.base_url, "planner", val);
              setLocalSelection(provider.base_url, "generator", val);
              setLocalSelection(provider.base_url, "gatherer", val);
            } else {
              setLocalSelection(provider.base_url, field, val);
            }
          }}
          disabled={!provider.models.length}
        >
          <SelectTrigger className="h-9 text-sm bg-muted/30">
            <SelectValue placeholder="Select model" />
          </SelectTrigger>
          <SelectContent>
            {models.map(model => (
              <SelectItem key={model} value={model} className="text-sm">
                <div className="flex items-center gap-2">
                  {model}
                  {provider.running.includes(model) && (
                    <span className="flex h-1.5 w-1.5 rounded-full bg-primary" title="Running" />
                  )}
                </div>
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>
    );
  };

  const renderLocalProvider = (
    provider: LocalProviderInfo | undefined,
    name: string,
    downloadUrl: string,
    filterTerm: string,
    setFilter: (val: string) => void,
  ) => {
    if (localProvidersStatus === "loading") {
      return (
        <div className="flex flex-col items-center justify-center py-12">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <p className="mt-3 text-sm text-muted-foreground">Scanning for {name}...</p>
        </div>
      );
    }

    if (!provider || provider.status === "error") {
      return (
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <Server className="h-10 w-10 text-muted-foreground/50" />
          <h3 className="mt-4 font-medium">{name} not detected</h3>
          <p className="mt-1 text-sm text-muted-foreground max-w-xs">
            Start {name} on your machine, then click Refresh
          </p>
          <div className="flex gap-2 mt-4">
            <Button variant="outline" size="sm" onClick={refreshLocalProviders}>
              Refresh
            </Button>
            <Button variant="outline" size="sm" asChild>
              <a href={downloadUrl} target="_blank" rel="noreferrer">
                Download <ExternalLink className="ml-1 h-3 w-3" />
              </a>
            </Button>
          </div>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
          <div>
            <p className="font-medium">{provider.name}</p>
            <p className="text-xs text-muted-foreground font-mono">{provider.base_url}</p>
          </div>
          <div className="flex items-center gap-2">
            <span className="flex h-2 w-2 rounded-full bg-primary" />
            <span className="text-xs text-muted-foreground">{provider.models.length} models</span>
          </div>
        </div>

        <div className="flex gap-2">
          <Input
            value={filterTerm}
            onChange={e => setFilter(e.target.value)}
            placeholder="Search models"
            className="h-9 text-sm"
          />
          <Button variant="outline" size="sm" onClick={refreshLocalProviders}>
            Refresh
          </Button>
        </div>

        <div className="flex items-center justify-between">
          <div className="text-xs text-muted-foreground font-mono">{provider.base_url}</div>
          <div className="flex items-center gap-2 text-xs text-muted-foreground">{filterModels(provider.models, filterTerm).length} shown</div>
        </div>

        {renderModelSelect(provider, "all", "Use for all roles", filterTerm)}

        <div className="grid gap-3 sm:grid-cols-3 pt-2 border-t border-border/50">
          {renderModelSelect(provider, "planner", "Planner", filterTerm)}
          {renderModelSelect(provider, "gatherer", "Gatherer", filterTerm)}
          {renderModelSelect(provider, "generator", "Generator", filterTerm)}
        </div>

        <Button
          onClick={() => applyLocalProvider(provider)}
          disabled={!provider.models.length || isSavingConfig}
          className="w-full"
        >
          <CheckCircle2 className="mr-2 h-4 w-4" />
          Apply {provider.name}
        </Button>
      </div>
    );
  };

  const renderOllamaCloud = () => {
    const filtered = filterModels(ollamaCloudModels, ollamaCloudFilter);

    return (
      <div className="space-y-4">
        <div className="flex items-start gap-3 p-3 rounded-lg bg-primary/10 border border-primary/20">
          <Cloud className="h-5 w-5 text-primary mt-0.5" />
          <div className="text-sm">
            <p className="font-medium text-primary">Ollama Cloud</p>
            <p className="text-muted-foreground text-xs mt-1">
              Run large models without local GPU. Free tier available with no data retention.
            </p>
          </div>
        </div>

        <form
          className="space-y-2"
          onSubmit={(event) => {
            event.preventDefault();
            void fetchOllamaCloudModels();
          }}
        >
          <Label htmlFor="ollama-cloud-api-key" className="text-xs font-bold uppercase tracking-wider">API Key</Label>
          <div className="flex gap-2">
            <Input
              id="ollama-cloud-api-key"
              type="password"
              value={ollamaCloudApiKey}
              onChange={(e) => setOllamaCloudApiKey(e.target.value)}
              autoComplete="off"
              placeholder="Enter your Ollama API key"
              className="flex-1"
            />
            <Button variant="outline" size="sm" asChild>
              <a href="https://ollama.com/settings/keys" target="_blank" rel="noreferrer">
                Get Key <ExternalLink className="ml-1 h-3 w-3" />
              </a>
            </Button>
          </div>
        </form>

        {ollamaCloudError && (
          <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 text-sm text-destructive">
            {ollamaCloudError}
          </div>
        )}

        {ollamaCloudLoading ? (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : ollamaCloudModels.length > 0 ? (
          <>
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <Input
                value={ollamaCloudFilter}
                onChange={e => setOllamaCloudFilter(e.target.value)}
                placeholder="Search cloud models"
                className="h-9 text-sm"
              />
              <div className="text-xs text-muted-foreground">
                {filtered.length} / {ollamaCloudModels.length} shown
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                Use for all roles
              </Label>
              <Select
                value={ollamaCloudSelection.planner}
                onValueChange={(val) => setOllamaCloudSelection({ planner: val, generator: val, gatherer: val })}
              >
                <SelectTrigger className="h-9 text-sm bg-muted/30">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent className="max-h-[250px]">
                  {filtered.slice(0, 50).map(model => (
                    <SelectItem key={model} value={model} className="text-sm">
                      {model}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-3 sm:grid-cols-3 pt-2 border-t border-border/50">
              <div className="space-y-1.5">
                <Label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Planner</Label>
                <Select
                  value={ollamaCloudSelection.planner}
                  onValueChange={(val) => setOllamaCloudSelection(prev => ({ ...prev, planner: val }))}
                >
                  <SelectTrigger className="h-9 text-sm bg-muted/30">
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent className="max-h-[250px]">
                    {filtered.slice(0, 50).map(model => (
                      <SelectItem key={model} value={model} className="text-sm">
                        {model}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Gatherer</Label>
                <Select
                  value={ollamaCloudSelection.gatherer}
                  onValueChange={(val) => setOllamaCloudSelection(prev => ({ ...prev, gatherer: val }))}
                >
                  <SelectTrigger className="h-9 text-sm bg-muted/30">
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent className="max-h-[250px]">
                    {filtered.slice(0, 50).map(model => (
                      <SelectItem key={model} value={model} className="text-sm">
                        {model}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Generator</Label>
                <Select
                  value={ollamaCloudSelection.generator}
                  onValueChange={(val) => setOllamaCloudSelection(prev => ({ ...prev, generator: val }))}
                >
                  <SelectTrigger className="h-9 text-sm bg-muted/30">
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent className="max-h-[250px]">
                    {filtered.slice(0, 50).map(model => (
                      <SelectItem key={model} value={model} className="text-sm">
                        {model}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-4 text-sm text-muted-foreground">
            Enter API key and click refresh to load models
          </div>
        )}

        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={fetchOllamaCloudModels}
            disabled={ollamaCloudLoading}
          >
            {ollamaCloudLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Refresh Models"}
          </Button>
          <Button
            onClick={() => {
              applyOllamaCloud();
            }}
            disabled={isSavingConfig}
            className="flex-1"
          >
            <CheckCircle2 className="mr-2 h-4 w-4" />
            Apply Ollama Cloud
          </Button>
        </div>
      </div>
    );
  };

  const renderOpenRouter = () => {
    const filtered = filterOpenRouterModels(openRouterModels, openRouterFilter);

    return (
      <div className="space-y-4">
        <form
          className="space-y-2"
          onSubmit={(event) => {
            event.preventDefault();
            void fetchOpenRouterModels();
          }}
        >
          <Label htmlFor="openrouter-api-key" className="text-xs font-bold uppercase tracking-wider">API Key</Label>
          <div className="flex gap-2">
            <Input
              id="openrouter-api-key"
              type="password"
              value={openRouterApiKey}
              onChange={(e) => setOpenRouterApiKey(e.target.value)}
              autoComplete="off"
              placeholder="sk-or-v1-..."
              className="flex-1"
            />
            <Button variant="outline" size="sm" asChild>
              <a href="https://openrouter.ai/keys" target="_blank" rel="noreferrer">
                Get Key <ExternalLink className="ml-1 h-3 w-3" />
              </a>
            </Button>
          </div>
        </form>

        {openRouterLoading ? (
          <div className="flex items-center justify-center py-6">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
          </div>
        ) : openRouterModels.length > 0 ? (
          <>
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
              <Input
                value={openRouterFilter}
                onChange={e => setOpenRouterFilter(e.target.value)}
                placeholder="Search OpenRouter models"
                className="h-9 text-sm"
              />
              <div className="text-xs text-muted-foreground">
                {filtered.length} / {openRouterModels.length} shown
              </div>
            </div>

            <div className="space-y-1.5">
              <Label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">
                Use for all roles
              </Label>
              <Select
                value={openRouterSelection.planner}
                onValueChange={(val) => setOpenRouterSelection({ planner: val, generator: val, gatherer: val })}
              >
                <SelectTrigger className="h-9 text-sm bg-muted/30">
                  <SelectValue placeholder="Select a model" />
                </SelectTrigger>
                <SelectContent className="max-h-[250px]">
                  {filtered.slice(0, 50).map(model => (
                    <SelectItem key={model.id} value={model.id} className="text-sm">
                      {model.name || model.id}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid gap-3 sm:grid-cols-3 pt-2 border-t border-border/50">
              <div className="space-y-1.5">
                <Label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Planner</Label>
                <Select
                  value={openRouterSelection.planner}
                  onValueChange={(val) => setOpenRouterSelection(prev => ({ ...prev, planner: val }))}
                >
                  <SelectTrigger className="h-9 text-sm bg-muted/30">
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent className="max-h-[250px]">
                    {filtered.slice(0, 50).map(model => (
                      <SelectItem key={model.id} value={model.id} className="text-sm">
                        {model.name || model.id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Gatherer</Label>
                <Select
                  value={openRouterSelection.gatherer}
                  onValueChange={(val) => setOpenRouterSelection(prev => ({ ...prev, gatherer: val }))}
                >
                  <SelectTrigger className="h-9 text-sm bg-muted/30">
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent className="max-h-[250px]">
                    {filtered.slice(0, 50).map(model => (
                      <SelectItem key={model.id} value={model.id} className="text-sm">
                        {model.name || model.id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-1.5">
                <Label className="text-[10px] font-bold uppercase tracking-wider text-muted-foreground">Generator</Label>
                <Select
                  value={openRouterSelection.generator}
                  onValueChange={(val) => setOpenRouterSelection(prev => ({ ...prev, generator: val }))}
                >
                  <SelectTrigger className="h-9 text-sm bg-muted/30">
                    <SelectValue placeholder="Select" />
                  </SelectTrigger>
                  <SelectContent className="max-h-[250px]">
                    {filtered.slice(0, 50).map(model => (
                      <SelectItem key={model.id} value={model.id} className="text-sm">
                        {model.name || model.id}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-4 text-sm text-muted-foreground">
            Enter API key and click refresh to load models
          </div>
        )}

        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={fetchOpenRouterModels}
            disabled={openRouterLoading}
          >
            {openRouterLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : "Refresh Models"}
          </Button>
          <Button
            onClick={applyOpenRouter}
            disabled={!openRouterSelection.planner || isSavingConfig}
            className="flex-1"
          >
            <CheckCircle2 className="mr-2 h-4 w-4" />
            Apply OpenRouter
          </Button>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Provider Carousel */}
      <Card>
        <div className="flex items-center justify-between border-b border-border/50 px-4 py-3">
          <Button variant="ghost" size="icon" onClick={goPrev} className="h-8 w-8">
            <ChevronLeft className="h-4 w-4" />
          </Button>
          
          <div className="flex gap-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                  activeTab === tab.id
                    ? "bg-primary text-primary-foreground"
                    : "text-muted-foreground hover:bg-muted"
                }`}
              >
                <span className={activeTab === tab.id ? "" : tab.color}>{tab.icon}</span>
                {tab.label}
              </button>
            ))}
          </div>
          
          <Button variant="ghost" size="icon" onClick={goNext} className="h-8 w-8">
            <ChevronRight className="h-4 w-4" />
          </Button>
        </div>

        <CardContent className="pt-6">
          {activeTab === "ollama" &&
            renderLocalProvider(ollamaProvider, "Ollama", "https://ollama.com/download", ollamaFilter, setOllamaFilter)}
          {activeTab === "ollama_cloud" && renderOllamaCloud()}
          {activeTab === "lmstudio" &&
            renderLocalProvider(lmstudioProvider, "LM Studio", "https://lmstudio.ai/", lmstudioFilter, setLmstudioFilter)}
          {activeTab === "openrouter" && renderOpenRouter()}
        </CardContent>
      </Card>

      {/* RAGFuzz Integration - Simple Status */}
      <Card>
        <CardContent className="py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-primary/10">
                <Shield className="h-4 w-4 text-primary" />
              </div>
              <div>
                <p className="font-medium text-sm">RAGFuzz Integration</p>
                <p className="text-xs text-muted-foreground">
                  {ragfuzzStatus?.ragfuzz_enabled
                    ? "Ready - Start RAGFuzz and connect to this API"
                    : "Checking connection..."}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              {ragfuzzStatus?.ragfuzz_enabled && (
                <span className="flex h-2 w-2 rounded-full bg-primary" />
              )}
              <code className="text-xs bg-muted px-2 py-1 rounded">
                ragfuzz connect {typeof window !== "undefined" ? window.location.origin : "http://localhost:8000"}
              </code>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

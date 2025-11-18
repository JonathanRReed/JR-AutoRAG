import { type ChangeEvent, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import type { AppConfig, LocalProviderInfo, ProviderConfig as ProviderConfigType, ProviderProfile, RetrievalDefaults, RoleSelection } from "@/types";

interface ProviderConfigProps {
  config: AppConfig | null;
  setConfig: React.Dispatch<React.SetStateAction<AppConfig | null>>;
  selectedProfile: string;
  handleSelectProfile: (name: string) => void;
  newProfileName: string;
  setNewProfileName: (name: string) => void;
  handleAddProfile: () => void;
  handleDiscoverModels: () => void;
  isSavingConfig: boolean;
  handleSaveConfig: () => void;
  refreshAll: () => void;
  localProviders: LocalProviderInfo[];
  localProvidersStatus: "idle" | "loading" | "ready" | "error";
  refreshLocalProviders: () => void;
  localSelections: Record<string, RoleSelection>;
  setLocalSelection: (baseUrl: string, field: keyof RoleSelection, value: string) => void;
  applyLocalProvider: (provider: LocalProviderInfo) => void;
  modelOptions: string[];
}

export function ProviderConfig({
  config,
  setConfig,
  selectedProfile,
  handleSelectProfile,
  newProfileName,
  setNewProfileName,
  handleAddProfile,
  handleDiscoverModels,
  isSavingConfig,
  handleSaveConfig,
  refreshAll,
  localProviders,
  localProvidersStatus,
  refreshLocalProviders,
  localSelections,
  setLocalSelection,
  applyLocalProvider,
  modelOptions,
}: ProviderConfigProps) {
  const profileOptions = useMemo(() => {
    const names = new Set<string>([selectedProfile]);
    (config?.provider_profiles ?? []).forEach(profile => names.add(profile.name));
    return Array.from(names);
  }, [config, selectedProfile]);

  const configProvider = config?.provider;
  const retrieval = config?.retrieval;

  const updateProvider = (field: keyof ProviderConfigType, value: string) => {
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

  const handleNumber = (event: ChangeEvent<HTMLInputElement>, field: keyof RetrievalDefaults) => {
    updateRetrieval(field, Number(event.target.value));
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
    <>
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
      <datalist id="modelOptions">
        {modelOptions.map(model => (
          <option key={model} value={model} />
        ))}
      </datalist>
    </>
  );
}

import { type ChangeEvent, useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { CheckCircle2, Loader2, RadioTower, Rocket, Server, Settings2, TriangleAlert, Info } from "lucide-react";
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

function InlineHint({ label, detail }: { label: string; detail: string }) {
  return (
    <span
      className="inline-flex items-center gap-2 rounded-full bg-muted px-2.5 py-1 text-[11px] font-medium text-muted-foreground"
      title={detail}
    >
      <Info className="h-3.5 w-3.5 text-secondary-foreground" />
      <span className="truncate">{label}</span>
    </span>
  );
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

  const renderProviderModels = (provider: LocalProviderInfo, field: keyof RoleSelection | "all", label: string) => {
    const selection = localSelections[provider.base_url];
    const value = field === "all" ? (selection?.planner || "") : (selection?.[field] ?? "");

    return (
      <div className="space-y-1">
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
          <SelectTrigger className="h-8 text-xs bg-muted/30 hover:bg-muted/50 transition-colors w-full min-w-0 overflow-hidden">
            <div className="flex-1 truncate text-left">
              <SelectValue placeholder="Select model" />
            </div>
          </SelectTrigger>
          <SelectContent>
            {provider.models.map(model => {
              const isRunning = provider.running.includes(model);
              return (
                <SelectItem key={model} value={model} className="text-xs">
                  <div className="flex items-center gap-2">
                    {model}
                    {isRunning && (
                      <span className="flex h-1.5 w-1.5 rounded-full bg-primary" title="Currently running" />
                    )}
                  </div>
                </SelectItem>
              );
            })}
          </SelectContent>
        </Select>
      </div>
    );
  };

  return (
    <div className="space-y-8">
      <div className="grid gap-6 grid-cols-1 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] xl:items-stretch">
        {/* Active Profile & Core Settings */}
        <Card className="flex h-full flex-col overflow-hidden">
          <CardHeader className="bg-muted/20">
            <CardTitle className="flex items-center gap-3">
              <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                <Settings2 className="h-5 w-5 text-primary" />
              </div>
              <div>
                <span className="flex items-center gap-2">
                  Provider Settings
                  <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground uppercase">
                    {selectedProfile || "default"}
                  </span>
                  <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground uppercase">
                    Active
                  </span>
                </span>
              </div>
            </CardTitle>
            <CardDescription>
              Configure your AI backend and manage reusable profiles.
            </CardDescription>
            <div className="mt-3 flex flex-wrap gap-2">
              <InlineHint label="Step 1: Select models" detail="Planner = break down queries; Gatherer = fetch; Generator = answer." />
              <InlineHint label="Step 2: Save profile" detail="Keep stable configs per environment (local vs cloud)." />
              <InlineHint label="Step 3: Apply detected runtime" detail="Use auto-detected Ollama / LM Studio models instantly." />
            </div>
          </CardHeader>
          <CardContent className="flex h-full flex-1 flex-col space-y-8 pt-8 overflow-y-auto">
            <div className="grid gap-6 grid-cols-1 2xl:grid-cols-[240px_minmax(0,1fr)]">
              {/* Profiles Sidebar */}
              <div className="space-y-6 rounded-lg border border-border/60 bg-muted/10 p-4">
                <div className="space-y-2">
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
                </div>

                <div className="space-y-3">
                  <Label htmlFor="newProfile">Save current as...</Label>
                  <div className="flex flex-col gap-2">
                    <Input
                      id="newProfile"
                      value={newProfileName}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => setNewProfileName(e.target.value)}
                      placeholder="Profile name"
                      className="h-8 text-xs"
                    />
                    <div className="grid grid-cols-2 gap-2">
                      <Button variant="default" size="sm" onClick={handleAddProfile} className="text-xs">
                        Save Profile
                      </Button>
                      <Button variant="outline" size="sm" onClick={handleDiscoverModels} className="text-xs">
                        Scan Models
                      </Button>
                    </div>
                  </div>
                </div>

                {config?.provider_profiles && config.provider_profiles.length > 0 && (
                  <div className="pt-2">
                    <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Saved Profiles</p>
                    <ul className="mt-3 space-y-2">
                      {config.provider_profiles.map(profile => (
                        <li key={profile.name} className="flex flex-col rounded-md border border-border/60 bg-card p-2 text-xs">
                          <span className="font-bold text-foreground">{profile.name}</span>
                          <span className="text-muted-foreground truncate">{profile.provider.name}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Config Fields */}
              <div className="grid gap-6">
                <div className="grid gap-4 sm:grid-cols-2">
                  <div className="space-y-2">
                    <Label htmlFor="providerName" className="text-xs">Provider Name</Label>
                    <Input
                      id="providerName"
                      value={configProvider?.name ?? ""}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateProvider("name", e.target.value)}
                      placeholder="e.g. OpenAI, Ollama"
                      title="Display name for this provider profile"
                    />
                  </div>
                  <div className="space-y-2 sm:col-span-2">
                    <Label htmlFor="providerUrl" className="text-xs">Base URL</Label>
                    <Input
                      id="providerUrl"
                      value={configProvider?.base_url ?? ""}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateProvider("base_url", e.target.value)}
                      placeholder="http://localhost:11434"
                      title="Endpoint for your runtime (Ollama/LM Studio/self-hosted)"
                    />
                  </div>
                </div>

                <div className="grid gap-4 sm:grid-cols-2 2xl:grid-cols-3">
                  <div className="space-y-2">
                    <Label htmlFor="plannerModel" className="text-xs">Planner Model</Label>
                    <Input
                      id="plannerModel"
                      list="modelOptions"
                      value={configProvider?.planner_model ?? ""}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateProvider("planner_model", e.target.value)}
                      placeholder="llama3-8b"
                      title="Planner: decomposes questions and proposes searches."
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="generatorModel" className="text-xs">Generator Model</Label>
                    <Input
                      id="generatorModel"
                      list="modelOptions"
                      value={configProvider?.generator_model ?? ""}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateProvider("generator_model", e.target.value)}
                      placeholder="llama3-8b"
                      title="Generator: crafts the final response."
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="gathererModel" className="text-xs">Gatherer Model</Label>
                    <Input
                      id="gathererModel"
                      list="modelOptions"
                      value={configProvider?.gatherer_model ?? ""}
                      onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateProvider("gatherer_model", e.target.value)}
                      placeholder="llama3-8b"
                      title="Gatherer: executes searches and returns source snippets."
                    />
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="apiKey" className="text-xs font-bold uppercase tracking-wider">API Key (if required)</Label>
                  <Input
                    id="apiKey"
                    type="password"
                    value={configProvider?.api_key ?? ""}
                    onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateProvider("api_key", e.target.value)}
                    placeholder="sk-..."
                    className="max-w-md"
                    title="Stored locally; used when your provider requires a key."
                  />
                </div>

                <div className="flex items-center gap-4 pt-4">
                  <Button
                    onClick={handleSaveConfig}
                    disabled={isSavingConfig}
                    className="bg-primary text-primary-foreground px-8"
                  >
                    {isSavingConfig ? "Saving..." : "Save Configuration"}
                  </Button>
                  <Button variant="ghost" onClick={refreshAll} className="text-muted-foreground">
                    Reset Changes
                  </Button>
                  <span className="text-xs text-muted-foreground">
                    Profile: {selectedProfile || "default"}
                  </span>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Local Runtimes */}
        <Card className="flex h-full flex-col overflow-hidden">
          <CardHeader className="bg-muted/20">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                  <Server className="h-5 w-5 text-primary" />
                </div>
                <div>
                  <CardTitle className="flex items-center gap-2">
                    Auto-Detected Providers
                    {localProviders.length > 0 && (
                      <span className="flex h-5 w-5 items-center justify-center rounded-full bg-muted text-[10px] font-semibold text-muted-foreground">
                        {localProviders.length}
                      </span>
                    )}
                  </CardTitle>
                  <CardDescription>
                    Click to apply settings from running Ollama or LM Studio instances.
                  </CardDescription>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <Button asChild size="sm" variant="outline" className="h-8 text-xs">
                      <a href="https://ollama.com/download" target="_blank" rel="noreferrer">
                        Download Ollama
                      </a>
                    </Button>
                    <Button asChild size="sm" variant="outline" className="h-8 text-xs">
                      <a href="https://lmstudio.ai/" target="_blank" rel="noreferrer">
                        Download LM Studio
                      </a>
                    </Button>
                  </div>
                </div>
              </div>
              <Button
                variant="outline"
                size="sm"
                onClick={refreshLocalProviders}
                disabled={localProvidersStatus === "loading"}
                className="gap-2"
              >
                {localProvidersStatus === "loading" ? (
                  <><Loader2 className="h-3.5 w-3.5 animate-spin" /> Scanning...</>
                ) : (
                  "Refresh"
                )}
              </Button>
            </div>
          </CardHeader>
          <CardContent className="flex h-full flex-1 flex-col pt-8 overflow-y-auto">
            <div className="flex h-full flex-col space-y-4">
              {localProviders.length === 0 ? (
                <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border/70 bg-muted/10 p-12 text-center">
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
                    <RadioTower className="h-6 w-6 text-muted-foreground" />
                  </div>
                  <h3 className="mt-4 font-medium text-foreground">No local runtimes detected</h3>
                  <p className="mt-1 max-w-sm text-sm text-muted-foreground">
                    Start Ollama or LM Studio on your machine, then click Refresh to detect available models.
                  </p>
                </div>
              ) : (
                <div className="grid gap-4 grid-cols-1">
                  <div className="flex items-center justify-between rounded-lg border border-border/60 bg-muted/10 p-3 text-xs text-muted-foreground">
                    <div className="flex items-center gap-2">
                      <Info className="h-3.5 w-3.5" />
                      <span>Apply a detected runtime to auto-fill planner, gatherer, and generator.</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={refreshLocalProviders}
                        disabled={localProvidersStatus === "loading"}
                        className="h-8 text-xs"
                      >
                        {localProvidersStatus === "loading" ? "Scanning..." : "Rescan"}
                      </Button>
                      <Button
                        variant="secondary"
                        size="sm"
                        disabled={localProviders.length === 0 || isSavingConfig}
                        onClick={() => {
                          const first = localProviders[0];
                          if (first) {
                            void applyLocalProvider(first);
                          }
                        }}
                        className="h-8 text-xs"
                      >
                        Use detected models
                      </Button>
                    </div>
                  </div>
                  {localProviders.map(provider => {
                    const running = provider.running.filter(Boolean);
                    const models = provider.models;
                    const kind = provider.kind;

                    return (
                      <div
                        key={provider.base_url}
                        className="group flex flex-col justify-between rounded-lg border border-border/60 bg-card p-5 transition-colors"
                      >
                        <div className="space-y-4">
                          <div className="flex items-center justify-between">
                            <div>
                              <p className="font-bold text-foreground">{provider.name}</p>
                              <p className="text-[10px] text-muted-foreground font-mono">{provider.base_url}</p>
                            </div>
                            <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-bold text-muted-foreground uppercase">
                              {kind}
                            </span>
                          </div>

                          {provider.status === "error" && provider.error_message ? (
                            <div className="rounded-lg bg-destructive/15 p-2 text-[10px] text-destructive font-medium">
                              <TriangleAlert className="mr-1 inline h-3.5 w-3.5" />
                              {provider.error_message}
                            </div>
                          ) : (
                            <div className="space-y-3">
                              {/* Fast-set option */}
                              {renderProviderModels(provider, "all", "Use for all roles")}

                              <div className="flex flex-col gap-2 mt-2 pt-2 border-t border-border/50">
                                {renderProviderModels(provider, "planner", "Planner")}
                                {renderProviderModels(provider, "gatherer", "Gatherer")}
                                {renderProviderModels(provider, "generator", "Generator")}
                              </div>
                              <div className="flex items-center gap-2 text-[10px]">
                                <span className="h-1.5 w-1.5 rounded-full bg-primary" />
                                <span className="text-muted-foreground">
                                  {models.length} models, {provider.running.length} active
                                </span>
                              </div>
                            </div>
                          )}
                        </div>

                        <Button
                          onClick={() => void applyLocalProvider(provider)}
                          disabled={!models.length || isSavingConfig}
                          className="mt-6 w-full gap-2"
                        >
                          <CheckCircle2 className="h-4 w-4" />
                          Apply {provider.name}
                        </Button>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </CardContent>
        </Card>
      </div>

      <datalist id="modelOptions">
        {modelOptions.map(model => (
          <option key={model} value={model} />
        ))}
      </datalist>
    </div>
  );
}

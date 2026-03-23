import { useMemo } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Settings2, Info } from "lucide-react";
import type { AppConfig, DeploymentProfile, ProviderConfig as ProviderConfigType, SubsystemBackendConfig } from "@/types";

interface ProviderConfigProps {
  config: AppConfig | null;
  setConfig: React.Dispatch<React.SetStateAction<AppConfig | null>>;
  updateDeploymentProfile: (value: DeploymentProfile) => void;
  updateBackend: (subsystem: string, patch: Partial<SubsystemBackendConfig>) => void;
  selectedProfile: string;
  handleSelectProfile: (name: string) => void;
  newProfileName: string;
  setNewProfileName: (name: string) => void;
  handleAddProfile: () => void;
  handleDiscoverModels: () => void;
  isSavingConfig: boolean;
  handleSaveConfig: () => void;
  refreshAll: () => void;
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
  updateDeploymentProfile,
  updateBackend,
  selectedProfile,
  handleSelectProfile,
  newProfileName,
  setNewProfileName,
  handleAddProfile,
  handleDiscoverModels,
  isSavingConfig,
  handleSaveConfig,
  refreshAll,
  modelOptions,
}: ProviderConfigProps) {
  const profileOptions = useMemo(() => {
    const names = new Set<string>([selectedProfile]);
    (config?.provider_profiles ?? []).forEach(profile => names.add(profile.name));
    return Array.from(names);
  }, [config, selectedProfile]);

  const configProvider = config?.provider;
  const backendEntries = [
    ["document_parser", "Document Parser"],
    ["ocr", "OCR Backend"],
    ["llm", "LLM Lane"],
    ["memory", "Memory"],
    ["telemetry", "Telemetry"],
  ] as const;
  const ocrVisionModel = config?.backends?.ocr?.settings?.vision_model;

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

  return (
    <Card className="overflow-hidden">
      <CardHeader className="bg-muted/20">
        <CardTitle className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
            <Settings2 className="h-5 w-5 text-primary" />
          </div>
          <div>
            <span className="flex items-center gap-2">
              Advanced Provider Settings
              <span className="rounded-full bg-muted px-2 py-0.5 text-[10px] font-semibold text-muted-foreground uppercase">
                {selectedProfile || "default"}
              </span>
            </span>
          </div>
        </CardTitle>
        <CardDescription>
          Manual configuration for custom endpoints and saved profiles.
        </CardDescription>
        <div className="mt-3 flex flex-wrap gap-2">
          <InlineHint label="Profiles" detail="Save and switch between provider configurations." />
          <InlineHint label="Manual entry" detail="Enter custom URLs and model names." />
        </div>
      </CardHeader>
      <CardContent className="pt-6 space-y-6">
        <div className="grid gap-6 lg:grid-cols-[240px_1fr]">
          {/* Profiles Sidebar */}
          <div className="space-y-4 rounded-lg border border-border/60 bg-muted/10 p-4">
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

            <div className="space-y-2">
              <Label htmlFor="newProfile">Save current as...</Label>
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

            {config?.provider_profiles && config.provider_profiles.length > 0 && (
              <div className="pt-2 border-t border-border/50">
                <p className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Saved Profiles</p>
                <ul className="mt-2 space-y-1">
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
          <div className="space-y-4">
            <div className="grid gap-4 sm:grid-cols-2">
              <div className="space-y-2">
                <Label htmlFor="deploymentProfile" className="text-xs">Deployment Profile</Label>
                <Select
                  value={config?.deployment_profile ?? "local_only"}
                  onValueChange={(value) => updateDeploymentProfile(value as DeploymentProfile)}
                >
                  <SelectTrigger id="deploymentProfile">
                    <SelectValue placeholder="Select deployment mode" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="local_only">Local Only</SelectItem>
                    <SelectItem value="hybrid">Hybrid</SelectItem>
                    <SelectItem value="cloud_accelerated">Cloud Accelerated</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="providerName" className="text-xs">Provider Name</Label>
                <Input
                  id="providerName"
                  value={configProvider?.name ?? ""}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateProvider("name", e.target.value)}
                  placeholder="e.g. OpenAI, Ollama"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="providerUrl" className="text-xs">Base URL</Label>
                <Input
                  id="providerUrl"
                  value={configProvider?.base_url ?? ""}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateProvider("base_url", e.target.value)}
                  placeholder="http://localhost:11434"
                />
              </div>
            </div>

            <div className="grid gap-4 sm:grid-cols-3">
              <div className="space-y-2">
                <Label htmlFor="plannerModel" className="text-xs">Planner Model</Label>
                <Input
                  id="plannerModel"
                  list="modelOptions"
                  value={configProvider?.planner_model ?? ""}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateProvider("planner_model", e.target.value)}
                  placeholder="llama3-8b"
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
                />
              </div>
            </div>

            <div className="space-y-2">
              <Label htmlFor="apiKey" className="text-xs">API Key (if required)</Label>
              <Input
                id="apiKey"
                type="password"
                value={configProvider?.api_key ?? ""}
                onChange={(e: React.ChangeEvent<HTMLInputElement>) => updateProvider("api_key", e.target.value)}
                placeholder="sk-..."
                className="max-w-md"
              />
            </div>

            <div className="space-y-3 rounded-lg border border-border/60 bg-muted/10 p-4">
              <div>
                <p className="text-sm font-medium text-foreground">Local-First Backend Lanes</p>
                <p className="text-xs text-muted-foreground">Choose the active backend class per subsystem.</p>
              </div>
              <div className="grid gap-4 sm:grid-cols-2">
                {backendEntries.map(([key, label]) => {
                  const backend = config?.backends?.[key];
                  return (
                    <div key={key} className="space-y-2">
                      <Label htmlFor={`backend-${key}`} className="text-xs">{label}</Label>
                      <Input
                        id={`backend-${key}`}
                        value={backend?.backend_id ?? ""}
                        onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                          updateBackend(key, { backend_id: e.target.value, label: backend?.label ?? label })
                        }
                        placeholder={`${key}.local.default`}
                        className="font-mono text-xs"
                      />
                      <p className="text-[10px] uppercase tracking-wide text-muted-foreground">
                        {backend?.capabilities?.mode ?? "local"} · {backend?.capabilities?.requires_network ? "networked" : "offline-capable"}
                      </p>
                    </div>
                  );
                })}
              </div>
              <div className="space-y-2">
                <Label htmlFor="ocrVisionModel" className="text-xs">OCR Vision Model</Label>
                <Input
                  id="ocrVisionModel"
                  value={typeof ocrVisionModel === "string" ? ocrVisionModel : ""}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                    updateBackend("ocr", {
                      settings: {
                        ...(config?.backends?.ocr?.settings ?? {}),
                        vision_model: e.target.value,
                      },
                    })
                  }
                  placeholder="Uses provider generator model if empty"
                  className="font-mono text-xs"
                />
              </div>
            </div>

            <div className="flex items-center gap-3 pt-2">
              <Button
                onClick={handleSaveConfig}
                disabled={isSavingConfig}
              >
                {isSavingConfig ? "Saving..." : "Save Configuration"}
              </Button>
              <Button variant="ghost" onClick={refreshAll}>
                Reset
              </Button>
            </div>
          </div>
        </div>
      </CardContent>

      <datalist id="modelOptions">
        {modelOptions.map(model => (
          <option key={model} value={model} />
        ))}
      </datalist>
    </Card>
  );
}

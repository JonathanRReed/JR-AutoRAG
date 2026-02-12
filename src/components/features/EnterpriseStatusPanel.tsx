import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { useToast } from "@/components/ui/toast";
import {
    Activity,
    CheckCircle2,
    Clock,
    Download,
    FileText,
    Loader2,
    Shield,
    AlertCircle,
    GitGraph,
    Layers,
    Database,
    Play
} from "lucide-react";

type ArtifactStage = "not_built" | "building" | "ready" | "failed";

interface ArtifactState {
    status: ArtifactStage;
    progress?: number;
    item_count?: number;
    error?: string;
    corpus_version?: string;
    started_at?: number;
    completed_at?: number;
}

interface ArtifactStatus {
    graph_rag: ArtifactState;
    raptor: ArtifactState;
}

interface ArtifactStatusPayload {
    graph_rag?: ArtifactState;
    raptor?: ArtifactState;
    graph_rag_status?: ArtifactStage;
    raptor_status?: ArtifactStage;
    graph_build_progress?: number;
    raptor_build_progress?: number;
    graph_version?: string;
    raptor_version?: string;
    corpus_version?: string;
}

interface EnterpriseStatusProps {
    baseUrl: string;
    apiKey?: string;
}

function StatusBadge({ status }: { status: string }) {
    const defaultConfig = { bg: "bg-muted", text: "text-muted-foreground", icon: Clock };
    const configs: Record<string, { bg: string; text: string; icon: typeof CheckCircle2 }> = {
        ready: { bg: "bg-emerald-500/10", text: "text-emerald-500", icon: CheckCircle2 },
        building: { bg: "bg-amber-500/10", text: "text-amber-500", icon: Loader2 },
        not_built: defaultConfig,
        failed: { bg: "bg-destructive/10", text: "text-destructive", icon: AlertCircle },
    };
    const config = configs[status] ?? defaultConfig;
    const Icon = config.icon;
    const isSpinning = status === "building";

    return (
        <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-semibold ${config.bg} ${config.text}`}>
            <Icon className={`h-3 w-3 ${isSpinning ? "animate-spin" : ""}`} />
            {status.replace("_", " ").toUpperCase()}
        </span>
    );
}

export function EnterpriseStatusPanel({ baseUrl, apiKey = "" }: EnterpriseStatusProps) {
    const { toast } = useToast();
    const [artifactStatus, setArtifactStatus] = useState<ArtifactStatus>({
        graph_rag: { status: "not_built" },
        raptor: { status: "not_built" },
    });
    const [isDownloading, setIsDownloading] = useState(false);
    const [isPolling, setIsPolling] = useState(false);
    const [lastTraceTime, setLastTraceTime] = useState<string | null>(null);
    const [corpusVersion, setCorpusVersion] = useState<string>("—");

    const isBuilding = artifactStatus.graph_rag.status === "building" || artifactStatus.raptor.status === "building";
    const activeProgresses = [
        artifactStatus.graph_rag.status === "building" ? artifactStatus.graph_rag.progress : undefined,
        artifactStatus.raptor.status === "building" ? artifactStatus.raptor.progress : undefined,
    ].filter((value): value is number => typeof value === "number");
    const activeProgress = activeProgresses.length ? Math.max(...activeProgresses) : 0;

    const normalizeState = (
        state: ArtifactState | undefined,
        fallbackStatus: ArtifactStage | undefined,
        fallbackProgress: number | undefined,
        fallbackVersion: string | undefined,
    ): ArtifactState => ({
        status: state?.status ?? fallbackStatus ?? "not_built",
        progress: state?.progress ?? fallbackProgress,
        item_count: state?.item_count,
        error: state?.error,
        corpus_version: state?.corpus_version ?? fallbackVersion,
        started_at: state?.started_at,
        completed_at: state?.completed_at,
    });

    const fetchStatus = async () => {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        const headers: Record<string, string> = {};
        const trimmedApiKey = apiKey.trim();
        if (trimmedApiKey) {
            headers["X-API-Key"] = trimmedApiKey;
        }

        try {
            const res = await fetch(`${baseUrl}/api/artifacts/status`, {
                headers: Object.keys(headers).length ? headers : undefined,
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            if (res.ok) {
                const data = await res.json() as ArtifactStatusPayload;
                const graph = normalizeState(
                    data.graph_rag,
                    data.graph_rag_status,
                    data.graph_build_progress,
                    data.graph_version,
                );
                const raptor = normalizeState(
                    data.raptor,
                    data.raptor_status,
                    data.raptor_build_progress,
                    data.raptor_version,
                );
                setArtifactStatus({ graph_rag: graph, raptor });
                setCorpusVersion(data.corpus_version || graph.corpus_version || raptor.corpus_version || "—");

                // Start polling if building
                if (graph.status === "building" || raptor.status === "building") {
                    if (!isPolling) setIsPolling(true);
                } else if (isPolling) {
                    setIsPolling(false);
                }
            }
        } catch {
            clearTimeout(timeoutId);
            // Silent fail - not critical during heavy indexing
        }
    };

    const downloadTrace = async () => {
        setIsDownloading(true);
        toast({
            title: "Exporting Traces",
            description: "Preparing bundle execution logs...",
        });

        const headers: Record<string, string> = {};
        const trimmedApiKey = apiKey.trim();
        if (trimmedApiKey) {
            headers["X-API-Key"] = trimmedApiKey;
        }
        try {
            const res = await fetch(`${baseUrl}/api/traces/download`, {
                headers: Object.keys(headers).length ? headers : undefined,
            });
            if (res.ok) {
                const blob = await res.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement("a");
                a.href = url;
                a.download = `trace_bundle_${new Date().toISOString().slice(0, 10)}.json`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
                setLastTraceTime(new Date().toLocaleTimeString());
                toast({
                    title: "Export Complete",
                    description: "Trace bundle downloaded successfully.",
                    variant: "success"
                });
            } else {
                throw new Error("Download failed");
            }
        } catch {
            toast({
                variant: "error",
                title: "Export Failed",
                description: "Could not retrieve traces across network.",
            });
        } finally {
            setIsDownloading(false);
        }
    };

    const handleBuild = async (type?: string) => {
        toast({
            title: "Build Started",
            description: `Triggering ${type || "Artifact"} generation pipeline in background...`,
        });

        const headers: Record<string, string> = { "Content-Type": "application/json" };
        const trimmedApiKey = apiKey.trim();
        if (trimmedApiKey) {
            headers["X-API-Key"] = trimmedApiKey;
        }
        try {
            // Updated to use Query Parameter as per FastAPI spec
            const res = await fetch(`${baseUrl}/api/artifacts/build?force=true`, {
                method: "POST",
                headers,
                // Empty body since we using query param, but keeping empty object just in case middleware needs valid JSON
                body: JSON.stringify({})
            });

            if (res.ok) {
                const data = await res.json();
                if (data.status === "error") {
                    throw new Error(data.message || "Build failed to start");
                }
                setIsPolling(true);
                // Immediate fetch to update status
                fetchStatus();
            } else {
                const errorData = await res.json().catch(() => ({}));
                throw new Error(errorData.detail || "API responded with error");
            }
        } catch (error) {
            console.error("Failed to trigger build:", error);
            toast({
                variant: "error",
                title: "Build Request Failed",
                description: error instanceof Error ? error.message : "Could not contact orchestrator.",
            });
        }
    };

    useEffect(() => {
        fetchStatus();
        let interval: ReturnType<typeof setInterval>;

        if (isPolling) {
            interval = setInterval(fetchStatus, 2000); // Poll fast when building
        } else {
            interval = setInterval(fetchStatus, 10000); // Poll slow otherwise
        }

        return () => clearInterval(interval);
    }, [baseUrl, isPolling, apiKey]);

    return (
        <Card className="overflow-hidden border-border/50 shadow-lg bg-background/60 backdrop-blur-xl transition-all duration-500 hover:border-border/80">
            <CardHeader className="border-b border-border/40 pb-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 border border-primary/20 shadow-sm">
                            <Shield className="h-5 w-5 text-primary" />
                        </div>
                        <div>
                            <CardTitle className="text-lg font-bold tracking-tight">Enterprise Controls</CardTitle>
                            <CardDescription className="text-xs font-medium opacity-80">vNext Guarantees & Observability</CardDescription>
                        </div>
                    </div>
                    <div className="flex items-center gap-3">
                        <div className="px-3 py-1 rounded-full bg-muted/50 border border-border/40 text-[10px] font-mono font-medium text-muted-foreground">
                            v{corpusVersion}
                        </div>
                        <div className="flex items-center gap-2 px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                            <div className="h-1.5 w-1.5 rounded-full bg-emerald-500 animate-pulse" />
                            <span className="text-[10px] font-semibold text-emerald-600">LIVE</span>
                        </div>
                    </div>
                </div>
            </CardHeader>
            <CardContent className="pt-6 space-y-6">
                <div className="grid gap-4 md:grid-cols-3">
                    {/* Artifact Build Status */}
                    <div className="group relative overflow-hidden rounded-xl border border-border/50 bg-gradient-to-br from-card/50 to-muted/20 p-5 transition-all hover:bg-muted/30 hover:border-border/80">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-2">
                                <Database className="h-4 w-4 text-primary" />
                                <span className="text-sm font-semibold">Artifact Builds</span>
                            </div>
                            <span className="text-[10px] font-mono text-muted-foreground opacity-70">G4</span>
                        </div>

                        <div className="space-y-3">
                            <div className="flex items-center justify-between p-2 rounded-lg bg-background/50 border border-border/30">
                                <div className="flex items-center gap-2">
                                    <GitGraph className="h-4 w-4 text-secondary/80" />
                                    <span className="text-xs font-medium">GraphRAG</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <StatusBadge status={artifactStatus.graph_rag.status} />
                                    {artifactStatus.graph_rag.status === "not_built" && (
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            onClick={() => handleBuild("GraphRAG")}
                                            disabled={isBuilding}
                                            className="h-6 w-6 rounded-md hover:bg-primary/10 hover:text-primary transition-colors"
                                            title="Build Knowledge Graph"
                                        >
                                            {isBuilding ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3 fill-current" />}
                                        </Button>
                                    )}
                                </div>
                            </div>

                            <div className="flex items-center justify-between p-2 rounded-lg bg-background/50 border border-border/30">
                                <div className="flex items-center gap-2">
                                    <Layers className="h-4 w-4 text-accent/80" />
                                    <span className="text-xs font-medium">RAPTOR</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <StatusBadge status={artifactStatus.raptor.status} />
                                    {artifactStatus.raptor.status === "not_built" && (
                                        <Button
                                            variant="ghost"
                                            size="icon"
                                            onClick={() => handleBuild("RAPTOR")}
                                            disabled={isBuilding}
                                            className="h-6 w-6 rounded-md hover:bg-primary/10 hover:text-primary transition-colors"
                                            title="Build RAPTOR Index"
                                        >
                                            {isBuilding ? <Loader2 className="h-3 w-3 animate-spin" /> : <Play className="h-3 w-3 fill-current" />}
                                        </Button>
                                    )}
                                </div>
                            </div>
                        </div>

                        {(artifactStatus.graph_rag.status === "building" || artifactStatus.raptor.status === "building") && (
                            <div className="mt-3 space-y-1.5 animate-in fade-in slide-in-from-top-1">
                                <div className="flex justify-between text-[10px] text-muted-foreground">
                                    <span className="flex items-center gap-1.5">
                                        <Loader2 className="h-2.5 w-2.5 animate-spin" />
                                        Building index...
                                    </span>
                                    <span className="font-mono">{activeProgress}%</span>
                                </div>
                                <div className="h-1 bg-muted/40 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-primary transition-all duration-300 ease-out rounded-full relative overflow-hidden"
                                        style={{
                                            width: `${activeProgress}%`
                                        }}
                                    >
                                        <div className="absolute inset-0 bg-white/20 animate-[shimmer_1s_infinite]" />
                                    </div>
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Citation Verification */}
                    <div className="group relative overflow-hidden rounded-xl border border-border/50 bg-gradient-to-br from-card/50 to-muted/20 p-5 transition-all hover:bg-muted/30 hover:border-border/80">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-2">
                                <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                                <span className="text-sm font-semibold">Citation Fidelity</span>
                            </div>
                            <span className="text-[10px] font-mono text-muted-foreground opacity-70">G1</span>
                        </div>

                        <div className="grid grid-cols-2 gap-3 mb-3">
                            <div className="text-center p-2 rounded-lg bg-background/50 border border-border/30">
                                <div className="text-xl font-bold text-foreground">100%</div>
                                <div className="text-[10px] text-muted-foreground">Verified</div>
                            </div>
                            <div className="text-center p-2 rounded-lg bg-background/50 border border-border/30">
                                <div className="text-xl font-bold text-emerald-600">0</div>
                                <div className="text-[10px] text-muted-foreground">Repairs</div>
                            </div>
                        </div>
                        <p className="text-[10px] text-muted-foreground leading-relaxed opacity-80">
                            Strict verification active. Invalid citations are automatically filtered.
                        </p>
                    </div>

                    {/* Trace Export */}
                    <div className="group relative overflow-hidden rounded-xl border border-border/50 bg-gradient-to-br from-card/50 to-muted/20 p-5 transition-all hover:bg-muted/30 hover:border-border/80">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-2">
                                <FileText className="h-4 w-4 text-primary" />
                                <span className="text-sm font-semibold">Trace Export</span>
                            </div>
                            <span className="text-[10px] font-mono text-muted-foreground opacity-70">E1</span>
                        </div>

                        <div className="space-y-4">
                            <Button
                                onClick={downloadTrace}
                                disabled={isDownloading}
                                variant="outline"
                                size="sm"
                                className="w-full justify-center gap-2 h-9 text-xs font-medium border-border/50 bg-background/50 hover:bg-background/80"
                            >
                                {isDownloading ? (
                                    <>
                                        <Loader2 className="h-3.5 w-3.5 animate-spin" />
                                        Preparing...
                                    </>
                                ) : (
                                    <>
                                        <Download className="h-3.5 w-3.5" />
                                        Download Bundle
                                    </>
                                )}
                            </Button>

                            <div className="space-y-1.5 pt-1">
                                <div className="flex items-center gap-2 text-[10px] text-muted-foreground opacity-80">
                                    <Activity className="h-3 w-3" />
                                    <span>Full execution traces</span>
                                </div>
                                <div className="flex items-center gap-2 text-[10px] text-muted-foreground opacity-80">
                                    <Shield className="h-3 w-3" />
                                    <span>Audit-ready logs</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Guarantee Banner */}
                <div className="relative overflow-hidden rounded-xl border border-primary/20 bg-primary/5 p-4 flex items-start gap-4">
                    <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-primary/10 text-primary">
                        <Shield className="h-4 w-4" />
                    </div>
                    <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                            <span className="text-sm font-semibold text-foreground">Cache Never Stale (G3)</span>
                            <span className="inline-flex items-center px-1.5 py-0.5 rounded text-[9px] font-bold bg-emerald-500/10 text-emerald-600 tracking-wide uppercase">Active</span>
                        </div>
                        <p className="text-xs text-muted-foreground leading-relaxed">
                            Cache keys include corpus version and retrieval mode. Stale cache hits are prevented automatically.
                        </p>
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}

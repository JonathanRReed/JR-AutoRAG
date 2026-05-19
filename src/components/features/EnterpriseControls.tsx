import React, { useEffect, useState } from 'react';
import { Database, Network, Layers, Loader2, AlertCircle } from 'lucide-react';

interface EnterpriseControlsProps {
    className?: string;
    onViewArtifact?: (type: 'graph_rag' | 'raptor') => void;
}

interface ArtifactState {
    status: "not_built" | "building" | "ready" | "failed";
    item_count?: number;
    error?: string;
}

interface ArtifactStatus {
    graph_rag: ArtifactState;
    raptor: ArtifactState;
}

export function EnterpriseControls({ className, onViewArtifact }: EnterpriseControlsProps) {
    const [status, setStatus] = useState<ArtifactStatus | null>(null);

    useEffect(() => {
        const fetchStatus = async () => {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);

            try {
                const res = await fetch("/api/artifacts/status", {
                    signal: controller.signal
                });
                clearTimeout(timeoutId);
                if (res.ok) {
                    const data = await res.json();
                    setStatus(data);
                }
            } catch (e) {
                clearTimeout(timeoutId);
                // Silently ignore abort errors during heavy indexing
                if (e instanceof Error && e.name !== 'AbortError') {
                    console.error("Failed to fetch artifact status", e);
                }
            }
        };

        fetchStatus();
        const interval = setInterval(fetchStatus, 10000); // Reduced frequency during heavy load
        return () => clearInterval(interval);
    }, []);

    const StatusBadge = ({ state }: { state?: ArtifactState }) => {
        if (!state || state.status === "not_built") {
            return (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">
                    <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground"></span>
                    NOT BUILT
                </span>
            );
        }
        if (state.status === "building") {
            return (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-secondary/10 px-2 py-0.5 text-[10px] text-foreground ring-1 ring-inset ring-secondary/20">
                    <Loader2 className="h-3 w-3 animate-spin" />
                    BUILDING
                </span>
            );
        }
        if (state.status === "failed") {
            return (
                <span className="inline-flex items-center gap-1.5 rounded-full bg-destructive/10 px-2 py-0.5 text-[10px] text-destructive ring-1 ring-inset ring-destructive/20" title={state.error}>
                    <AlertCircle className="h-3 w-3" />
                    FAILED
                </span>
            );
        }
        return (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary ring-1 ring-inset ring-primary/20">
                <span className="relative flex h-1.5 w-1.5">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-primary"></span>
                </span>
                ACTIVE ({state.item_count ?? 0})
            </span>
        );
    };

    return (
        <div className={`mt-4 rounded-xl border border-border bg-card/80 p-4 shadow-sm ${className}`}>
            <div className="mb-3 flex items-center justify-between">
                <div className="flex items-center gap-2">
                    <Database className="h-4 w-4 text-primary" />
                    <span className="text-sm font-medium text-foreground">Artifact Builds</span>
                </div>
                <span className="text-[10px] font-mono text-muted-foreground">G4</span>
            </div>

            <div className="space-y-2">
                {/* GraphRAG Row */}
                <div
                    onClick={() => status?.graph_rag.status !== "not_built" && onViewArtifact?.('graph_rag')}
                    className={`group flex items-center justify-between rounded-lg bg-muted/30 p-2 transition-all ${status?.graph_rag.status !== "not_built" ? "cursor-pointer hover:bg-muted/50" : "opacity-50"}`}
                >
                    <div className="flex items-center gap-2">
                        <Network className="h-3.5 w-3.5 text-primary" />
                        <div>
                            <p className="text-xs font-medium text-foreground">GraphRAG</p>
                        </div>
                    </div>
                    <StatusBadge state={status?.graph_rag} />
                </div>

                {/* RAPTOR Row */}
                <div
                    onClick={() => status?.raptor.status !== "not_built" && onViewArtifact?.('raptor')}
                    className={`group flex items-center justify-between rounded-lg bg-muted/30 p-2 transition-all ${status?.raptor.status !== "not_built" ? "cursor-pointer hover:bg-muted/50" : "opacity-50"}`}
                >
                    <div className="flex items-center gap-2">
                        <Layers className="h-3.5 w-3.5 text-muted-foreground" />
                        <div>
                            <p className="text-xs font-medium text-foreground">RAPTOR</p>
                        </div>
                    </div>
                    <StatusBadge state={status?.raptor} />
                </div>
            </div>
        </div>
    );
}

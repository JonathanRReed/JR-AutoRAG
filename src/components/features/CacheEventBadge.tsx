import React from "react";
import { Database, Zap, XCircle } from "lucide-react";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import type { CacheEvent } from "@/types";

interface CacheEventBadgeProps {
    event?: CacheEvent;
    fromCache?: boolean;
}

/**
 * Badge showing cache hit/miss status for a query result.
 */
export function CacheEventBadge({ event, fromCache }: CacheEventBadgeProps) {
    // Determine cache status
    const isHit = fromCache || event?.hit;

    if (!event && !fromCache) {
        return null; // No cache info to display
    }

    const reasonLabels: Record<string, string> = {
        not_found: "First-time query",
        expired: "Cache expired",
        version_mismatch: "Corpus changed",
    };

    const reason = event?.reason ? reasonLabels[event.reason] || event.reason : "";

    return (
        <TooltipProvider>
            <Tooltip>
                <TooltipTrigger asChild>
                    <div
                        className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-medium transition-colors ${isHit
                                ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20"
                                : "bg-muted text-muted-foreground border border-border/50"
                            }`}
                    >
                        {isHit ? (
                            <>
                                <Zap className="h-3 w-3" />
                                <span>Cache Hit</span>
                            </>
                        ) : (
                            <>
                                <Database className="h-3 w-3" />
                                <span>Cache Miss</span>
                            </>
                        )}
                    </div>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="max-w-[250px]">
                    <div className="space-y-1 text-xs">
                        <p className="font-medium">
                            {isHit ? "Result from cache" : "Fresh query execution"}
                        </p>
                        {reason && (
                            <p className="text-muted-foreground">Reason: {reason}</p>
                        )}
                        {event && (
                            <div className="text-muted-foreground space-y-0.5 pt-1 border-t border-border/50 mt-1">
                                {event.corpus_version && (
                                    <p>Corpus: v{event.corpus_version.slice(0, 8)}</p>
                                )}
                                {event.preset_id && (
                                    <p>Preset: {event.preset_id}</p>
                                )}
                            </div>
                        )}
                    </div>
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    );
}

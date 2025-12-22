import React, { useState } from "react";
import { CheckCircle2, AlertTriangle, XCircle, ChevronDown, ChevronUp, FileText } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { GroundingInfo } from "@/types";

interface GroundingBadgeProps {
    grounding?: GroundingInfo;
    chunks?: Array<{ id: string; title: string; snippet: string; score: number }>;
}

/**
 * Badge showing grounding status with expandable evidence drawer.
 * 
 * Implements P1.8: Grounding Visible in Answers
 */
export function GroundingBadge({ grounding, chunks = [] }: GroundingBadgeProps) {
    const [isExpanded, setIsExpanded] = useState(false);

    if (!grounding) {
        return null;
    }

    const { grounded, docs_used, citations_kept, chunks_dropped } = grounding;

    // Determine status color and icon
    const getStatusDisplay = () => {
        if (grounded && docs_used > 0 && citations_kept > 0) {
            return {
                icon: CheckCircle2,
                color: "text-emerald-600 dark:text-emerald-400",
                bgColor: "bg-emerald-500/10",
                borderColor: "border-emerald-500/20",
                label: "Grounded",
            };
        } else if (docs_used > 0) {
            return {
                icon: AlertTriangle,
                color: "text-amber-600 dark:text-amber-400",
                bgColor: "bg-amber-500/10",
                borderColor: "border-amber-500/20",
                label: "Partially Grounded",
            };
        } else {
            return {
                icon: XCircle,
                color: "text-red-600 dark:text-red-400",
                bgColor: "bg-red-500/10",
                borderColor: "border-red-500/20",
                label: "Not Grounded",
            };
        }
    };

    const status = getStatusDisplay();
    const StatusIcon = status.icon;

    return (
        <div className="space-y-2">
            {/* Main badge */}
            <div
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg border ${status.bgColor} ${status.borderColor}`}
            >
                <StatusIcon className={`h-4 w-4 ${status.color}`} />
                <span className={`text-sm font-medium ${status.color}`}>
                    {status.label}
                </span>

                {/* Stats */}
                <div className="flex items-center gap-2 ml-2 pl-2 border-l border-border/50 text-xs text-muted-foreground">
                    <span>{docs_used} docs</span>
                    <span>•</span>
                    <span>{citations_kept} citations</span>
                    {chunks_dropped > 0 && (
                        <>
                            <span>•</span>
                            <span className="text-amber-600 dark:text-amber-400">
                                {chunks_dropped} dropped
                            </span>
                        </>
                    )}
                </div>

                {/* Expand toggle */}
                {chunks.length > 0 && (
                    <Button
                        variant="ghost"
                        size="sm"
                        className="h-6 w-6 p-0 ml-1"
                        onClick={() => setIsExpanded(!isExpanded)}
                    >
                        {isExpanded ? (
                            <ChevronUp className="h-3 w-3" />
                        ) : (
                            <ChevronDown className="h-3 w-3" />
                        )}
                    </Button>
                )}
            </div>

            {/* Evidence drawer */}
            {isExpanded && chunks.length > 0 && (
                <div className="p-3 rounded-lg border border-border/60 bg-muted/30 space-y-2 animate-in slide-in-from-top-2 duration-200">
                    <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider flex items-center gap-1.5">
                        <FileText className="h-3 w-3" />
                        Evidence Used ({chunks.length})
                    </h4>
                    <div className="space-y-2 max-h-[200px] overflow-y-auto">
                        {chunks.slice(0, 5).map((chunk, idx) => (
                            <div
                                key={chunk.id || idx}
                                className="p-2 rounded border border-border/40 bg-background text-xs"
                            >
                                <div className="flex items-center justify-between mb-1">
                                    <span className="font-medium text-foreground truncate">
                                        [{idx + 1}] {chunk.title}
                                    </span>
                                    <span className="text-muted-foreground shrink-0 ml-2">
                                        {(chunk.score * 100).toFixed(0)}%
                                    </span>
                                </div>
                                <p className="text-muted-foreground line-clamp-2">
                                    {chunk.snippet?.slice(0, 150)}...
                                </p>
                            </div>
                        ))}
                        {chunks.length > 5 && (
                            <p className="text-xs text-muted-foreground text-center py-1">
                                + {chunks.length - 5} more chunks
                            </p>
                        )}
                    </div>
                </div>
            )}

            {/* No evidence response */}
            {grounding.no_evidence_response && (
                <div className="p-3 rounded-lg border border-amber-500/30 bg-amber-500/5 text-sm">
                    <p className="text-amber-700 dark:text-amber-300 mb-2">
                        {grounding.no_evidence_response.message}
                    </p>
                    {grounding.no_evidence_response.suggested_actions?.length > 0 && (
                        <div className="space-y-1">
                            <p className="text-xs font-medium text-muted-foreground">Suggested:</p>
                            <ul className="text-xs text-muted-foreground space-y-0.5">
                                {grounding.no_evidence_response.suggested_actions.map((action, i) => (
                                    <li key={i} className="flex items-start gap-1.5">
                                        <span>•</span>
                                        <span>{action.label}: {action.description}</span>
                                    </li>
                                ))}
                            </ul>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

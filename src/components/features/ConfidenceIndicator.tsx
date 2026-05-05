import { useMemo } from "react";
import { AlertTriangle, Shield, ShieldAlert, ShieldCheck, ShieldQuestion, ShieldX, Info } from "lucide-react";

interface ConfidenceIndicatorProps {
    /** Confidence score from 0 to 1 */
    confidence: number;
    /** Optional breakdown of confidence factors */
    factors?: {
        retrieval?: number;
        generation?: number;
        citation?: number;
        evidence?: number;
        coverage?: number;
    };
    /** Whether the answer passed hallucination check */
    hallucinationPass?: boolean;
    /** Whether evidence contract was satisfied */
    evidenceContractPass?: boolean;
    /** Whether the system abstained from answering */
    abstained?: boolean;
    /** Reason for abstention */
    abstentionReason?: string;
    /** Compact mode for inline display */
    compact?: boolean;
}

type ConfidenceLevel = "high" | "medium" | "low" | "abstained" | "unknown";

const CONFIDENCE_LEVELS: Record<ConfidenceLevel, {
    label: string;
    color: string;
    bgColor: string;
    Icon: typeof Shield;
}> = {
    high: {
        label: "High Confidence",
        color: "text-primary",
        bgColor: "bg-primary/10",
        Icon: ShieldCheck,
    },
    medium: {
        label: "Medium Confidence",
        color: "text-muted-foreground",
        bgColor: "bg-muted",
        Icon: Shield,
    },
    low: {
        label: "Low Confidence",
        color: "text-destructive",
        bgColor: "bg-destructive/10",
        Icon: ShieldAlert,
    },
    abstained: {
        label: "Insufficient Evidence",
        color: "text-primary",
        bgColor: "bg-primary/10",
        Icon: ShieldX,
    },
    unknown: {
        label: "Unknown",
        color: "text-muted-foreground",
        bgColor: "bg-muted",
        Icon: ShieldQuestion,
    },
};

function getConfidenceLevel(score: number): ConfidenceLevel {
    if (score >= 0.8) return "high";
    if (score >= 0.5) return "medium";
    if (score > 0) return "low";
    return "unknown";
}

export function ConfidenceIndicator({
    confidence,
    factors,
    hallucinationPass,
    evidenceContractPass,
    abstained = false,
    abstentionReason,
    compact = false,
}: ConfidenceIndicatorProps) {
    const level = useMemo(() => {
        if (abstained) return "abstained";
        return getConfidenceLevel(confidence);
    }, [confidence, abstained]);
    const config = CONFIDENCE_LEVELS[level];
    const Icon = config.Icon;
    const percentage = Math.round(confidence * 100);

    if (compact) {
        return (
            <div
                className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${config.bgColor} ${config.color}`}
                title={abstained ? `${config.label}: ${abstentionReason || "Insufficient evidence"}` : `${config.label}: ${percentage}%`}
            >
                <Icon className="h-3.5 w-3.5" />
                <span>{abstained ? "Abstained" : `${percentage}%`}</span>
            </div>
        );
    }

    return (
        <div className={`rounded-xl border border-border/40 p-4 bg-background/50 backdrop-blur-sm shadow-sm transition-all hover:border-border/60 ${config.color.replace('text-', 'border-').replace('500', '200')}/20`}>
            {/* Header */}
            <div className="flex items-center justify-between gap-3 mb-3">
                <div className="flex items-center gap-2">
                    <div className={`p-1.5 rounded-lg ${config.bgColor}`}>
                        <Icon className={`h-4 w-4 ${config.color}`} />
                    </div>
                    <span className="text-sm font-medium text-foreground/90">
                        {config.label}
                    </span>
                </div>
                <span className={`text-lg font-bold ${config.color} tabular-nums tracking-tight`}>
                    {abstained ? "N/A" : `${percentage}%`}
                </span>
            </div>

            {/* Abstention Banner */}
            {abstained && abstentionReason && (
                <div className="flex items-start gap-2 p-3 mb-4 rounded-lg bg-primary/5 border border-primary/20">
                    <AlertTriangle className="h-4 w-4 text-primary mt-0.5 flex-shrink-0" />
                    <p className="text-xs text-primary">
                        {abstentionReason}
                    </p>
                </div>
            )}

            {/* Confidence Bar (only shown when not abstained) */}
            {!abstained && (
                <div className="h-1.5 bg-muted/40 rounded-full overflow-hidden mb-4">
                    <div
                        className={`h-full rounded-full transition-all duration-1000 ease-out ${config.color.replace('text-', 'bg-')}`}
                        style={{ width: `${percentage}%` }}
                    />
                </div>
            )}

            {/* Quality Checks */}
            <div className="flex flex-wrap gap-2 mb-4">
                {hallucinationPass !== undefined && (
                    <div
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-medium border ${hallucinationPass
                            ? "bg-primary/5 text-primary border-primary/20"
                            : "bg-destructive/5 text-destructive border-destructive/20"
                            }`}
                    >
                        {hallucinationPass ? <ShieldCheck className="h-3 w-3" /> : <ShieldAlert className="h-3 w-3" />}
                        Hallucination Check
                    </div>
                )}
                {evidenceContractPass !== undefined && (
                    <div
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-medium border ${evidenceContractPass
                            ? "bg-primary/5 text-primary border-primary/20"
                            : "bg-muted text-muted-foreground border-border"
                            }`}
                    >
                        {evidenceContractPass ? <ShieldCheck className="h-3 w-3" /> : <ShieldQuestion className="h-3 w-3" />}
                        Evidence Contract
                    </div>
                )}
                {abstained && (
                    <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-medium border bg-primary/5 text-primary border-primary/20">
                        <ShieldX className="h-3 w-3" />
                        Abstained
                    </div>
                )}
            </div>

            {/* Factor Breakdown */}
            {factors && Object.keys(factors).length > 0 && (
                <div className="space-y-2 pt-3 border-t border-border/40">
                    <div className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                        <Info className="h-3 w-3 opacity-70" />
                        <span>{abstained ? "Evidence Quality" : "Confidence Factors"}</span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                        {[
                            ['Retrieval', factors.retrieval],
                            ['Generation', factors.generation],
                            ['Citations', factors.citation],
                            ['Evidence', factors.evidence],
                            ['Coverage', factors.coverage]
                        ].map(([label, value]) => {
                            // Clamp value to 0-1 range to prevent negative or >100% display
                            const clampedValue = value !== undefined ? Math.max(0, Math.min(1, Number(value))) : undefined;
                            return (
                                clampedValue !== undefined && (
                                    <div key={String(label)} className="flex justify-between items-center group">
                                        <span className="text-muted-foreground group-hover:text-foreground transition-colors">{label}</span>
                                        <div className="flex items-center gap-2">
                                            <div className="w-12 h-1 bg-muted/40 rounded-full overflow-hidden">
                                                <div
                                                    className={`h-full rounded-full transition-all duration-500 ${clampedValue > 0.7 ? 'bg-primary/70' : clampedValue > 0.4 ? 'bg-secondary' : 'bg-destructive/70'}`}
                                                    style={{ width: `${clampedValue * 100}%` }}
                                                />
                                            </div>
                                            <span className="font-mono font-medium text-foreground/80 tabular-nums">{Math.round(clampedValue * 100)}%</span>
                                        </div>
                                    </div>
                                )
                            )
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}

export default ConfidenceIndicator;

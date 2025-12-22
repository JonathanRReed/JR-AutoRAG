import { useMemo } from "react";
import { Shield, ShieldAlert, ShieldCheck, ShieldQuestion, Info } from "lucide-react";

interface ConfidenceIndicatorProps {
    /** Confidence score from 0 to 1 */
    confidence: number;
    /** Optional breakdown of confidence factors */
    factors?: {
        retrieval?: number;
        generation?: number;
        citation?: number;
        evidence?: number;
    };
    /** Whether the answer passed hallucination check */
    hallucinationPass?: boolean;
    /** Whether evidence contract was satisfied */
    evidenceContractPass?: boolean;
    /** Compact mode for inline display */
    compact?: boolean;
}

type ConfidenceLevel = "high" | "medium" | "low" | "unknown";

const CONFIDENCE_LEVELS: Record<ConfidenceLevel, {
    label: string;
    color: string;
    bgColor: string;
    Icon: typeof Shield;
}> = {
    high: {
        label: "High Confidence",
        color: "text-emerald-500",
        bgColor: "bg-emerald-500/10",
        Icon: ShieldCheck,
    },
    medium: {
        label: "Medium Confidence",
        color: "text-amber-500",
        bgColor: "bg-amber-500/10",
        Icon: Shield,
    },
    low: {
        label: "Low Confidence",
        color: "text-red-500",
        bgColor: "bg-red-500/10",
        Icon: ShieldAlert,
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
    compact = false,
}: ConfidenceIndicatorProps) {
    const level = useMemo(() => getConfidenceLevel(confidence), [confidence]);
    const config = CONFIDENCE_LEVELS[level];
    const Icon = config.Icon;
    const percentage = Math.round(confidence * 100);

    if (compact) {
        return (
            <div
                className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium ${config.bgColor} ${config.color}`}
                title={`${config.label}: ${percentage}%`}
            >
                <Icon className="h-3.5 w-3.5" />
                <span>{percentage}%</span>
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
                    {percentage}%
                </span>
            </div>

            {/* Confidence Bar */}
            <div className="h-1.5 bg-muted/40 rounded-full overflow-hidden mb-4">
                <div
                    className={`h-full rounded-full transition-all duration-1000 ease-out ${config.color.replace('text-', 'bg-')}`}
                    style={{ width: `${percentage}%` }}
                />
            </div>

            {/* Quality Checks */}
            <div className="flex flex-wrap gap-2 mb-4">
                {hallucinationPass !== undefined && (
                    <div
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-medium border ${hallucinationPass
                            ? "bg-emerald-500/5 text-emerald-600 border-emerald-500/20"
                            : "bg-red-500/5 text-red-600 border-red-500/20"
                            }`}
                    >
                        {hallucinationPass ? <ShieldCheck className="h-3 w-3" /> : <ShieldAlert className="h-3 w-3" />}
                        Hallucination Check
                    </div>
                )}
                {evidenceContractPass !== undefined && (
                    <div
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-medium border ${evidenceContractPass
                            ? "bg-emerald-500/5 text-emerald-600 border-emerald-500/20"
                            : "bg-amber-500/5 text-amber-600 border-amber-500/20"
                            }`}
                    >
                        {evidenceContractPass ? <ShieldCheck className="h-3 w-3" /> : <ShieldQuestion className="h-3 w-3" />}
                        Evidence Contract
                    </div>
                )}
            </div>

            {/* Factor Breakdown */}
            {factors && Object.keys(factors).length > 0 && (
                <div className="space-y-2 pt-3 border-t border-border/40">
                    <div className="flex items-center gap-1.5 text-[10px] font-medium text-muted-foreground uppercase tracking-wider">
                        <Info className="h-3 w-3 opacity-70" />
                        <span>Confidence Factors</span>
                    </div>
                    <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-xs">
                        {[
                            ['Retrieval', factors.retrieval],
                            ['Generation', factors.generation],
                            ['Citations', factors.citation],
                            ['Evidence', factors.evidence]
                        ].map(([label, value]) => (
                            value !== undefined && (
                                <div key={String(label)} className="flex justify-between items-center group">
                                    <span className="text-muted-foreground group-hover:text-foreground transition-colors">{label}</span>
                                    <div className="flex items-center gap-2">
                                        <div className="w-12 h-1 bg-muted/40 rounded-full overflow-hidden">
                                            <div
                                                className={`h-full rounded-full transition-all duration-500 ${Number(value) > 0.7 ? 'bg-emerald-500/50' : Number(value) > 0.4 ? 'bg-amber-500/50' : 'bg-red-500/50'}`}
                                                style={{ width: `${Number(value) * 100}%` }}
                                            />
                                        </div>
                                        <span className="font-mono font-medium text-foreground/80 tabular-nums">{Math.round(Number(value) * 100)}%</span>
                                    </div>
                                </div>
                            )
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default ConfidenceIndicator;

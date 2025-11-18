import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import type { TraceOut } from "@/types";

interface TraceLogProps {
    isEvaluating: boolean;
    handleEvaluation: () => void;
    evaluationSummary: string;
    traces: TraceOut[];
    formatNumber: (value?: number) => string;
}

export function TraceLog({
    isEvaluating,
    handleEvaluation,
    evaluationSummary,
    traces,
    formatNumber,
}: TraceLogProps) {
    return (
        <Card>
            <CardHeader>
                <CardTitle>Evaluation & Monitoring</CardTitle>
                <CardDescription>Spot regressions and inspect traces.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                    <Button disabled={isEvaluating} onClick={handleEvaluation}>
                        {isEvaluating ? "Running..." : "Run Quick Evaluation"}
                    </Button>
                    {evaluationSummary && <p className="text-sm text-muted-foreground">{evaluationSummary}</p>}
                </div>
                <div>
                    <p className="text-sm font-semibold">Traces ({traces.length})</p>
                    <div className="mt-2 max-h-64 space-y-3 overflow-y-auto rounded-lg border border-border p-3 text-sm text-muted-foreground">
                        {traces.map(trace => (
                            <div key={trace.id} className="border-b border-border/60 pb-2 last:border-b-0 last:pb-0">
                                <p className="font-medium text-foreground">{trace.prompt}</p>
                                <p>{trace.answer.slice(0, 140)}...</p>
                                <small className="text-xs text-muted-foreground">
                                    coverage {formatNumber(trace.metrics.coverage)} · tokens {formatNumber(trace.metrics.tokens)}
                                </small>
                            </div>
                        ))}
                        {traces.length === 0 && <p>No traces yet. Run a query to capture one.</p>}
                    </div>
                </div>
            </CardContent>
        </Card>
    );
}

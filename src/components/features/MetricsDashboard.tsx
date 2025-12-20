import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Activity, Database, Gauge, TrendingUp, Zap, Info } from "lucide-react";
import type { CacheStats } from "@/types";

interface MetricsDashboardProps {
    traces: Array<{
        id: string;
        prompt: string;
        metrics: Record<string, number | string>;
    }>;
    cacheStats?: CacheStats;
    onClearCache?: () => void;
    isClearingCache?: boolean;
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

const getMetricNumber = (value: number | string | undefined) =>
    typeof value === "number" && Number.isFinite(value) ? value : 0;

function StatCard({
    title,
    value,
    subtitle,
    trend,
    icon,
}: {
    title: string;
    value: string | number;
    subtitle?: string;
    trend?: "up" | "down" | "neutral";
    icon?: React.ReactNode;
}) {
    const trendStyles = {
        up: "text-primary",
        down: "text-destructive",
        neutral: "text-muted-foreground",
    };

    return (
        <div className="group rounded-lg border border-border/60 bg-card p-5 transition-colors">
            <div className="flex items-start justify-between">
                <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                    {title}
                </p>
                {icon && (
                    <span className="text-muted-foreground/60 group-hover:text-primary transition-colors">
                        {icon}
                    </span>
                )}
            </div>
            <p className="mt-2 text-3xl font-bold text-foreground">{value}</p>
            {subtitle && (
                <p className={`mt-1.5 text-xs font-medium ${trend ? trendStyles[trend] : "text-muted-foreground"}`}>
                    {subtitle}
                </p>
            )}
        </div>
    );
}

function PerformanceBar({ label, value, max }: { label: string; value: number; max: number }) {
    const percentage = Math.min((value / max) * 100, 100);
    const barColor = percentage > 80 ? "bg-primary" : percentage > 50 ? "bg-secondary" : "bg-destructive";

    return (
        <div className="space-y-1">
            <div className="flex justify-between text-xs">
                <span className="text-muted-foreground">{label}</span>
                <span className="font-medium text-foreground">{value.toFixed(0)}ms</span>
            </div>
            <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div
                    className={`h-full rounded-full transition-all duration-500 ${barColor}`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    );
}

export function MetricsDashboard({ traces, cacheStats, onClearCache, isClearingCache }: MetricsDashboardProps) {
    // Calculate aggregate metrics
    const totalQueries = traces.length;
    const avgLatency = totalQueries > 0
        ? traces.reduce((sum, t) => sum + getMetricNumber(t.metrics.duration_ms as number | string), 0) / totalQueries
        : 0;
    const avgCoverage = totalQueries > 0
        ? traces.reduce((sum, t) => sum + getMetricNumber(t.metrics.coverage as number | string), 0) / totalQueries
        : 0;

    // Cache hit rate
    const embeddingHits = cacheStats?.embeddings?.hits || 0;
    const embeddingMisses = cacheStats?.embeddings?.misses || 0;
    const embeddingTotal = embeddingHits + embeddingMisses;
    const embeddingHitRate = embeddingTotal > 0 ? (embeddingHits / embeddingTotal) * 100 : 0;

    const queryHits = cacheStats?.queries?.hits || 0;
    const queryMisses = cacheStats?.queries?.misses || 0;
    const queryTotal = queryHits + queryMisses;
    const queryHitRate = queryTotal > 0 ? (queryHits / queryTotal) * 100 : 0;

    // Recent queries for latency display
    const recentQueries = traces.slice(-5).reverse();

    return (
        <Card className="overflow-hidden">
            <CardHeader className="bg-muted/20">
                <CardTitle className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                        <Activity className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                        <span className="flex items-center gap-2">
                            System Metrics
                            <span className="relative flex h-2 w-2">
                                <span className="relative inline-flex h-2 w-2 rounded-full bg-primary" />
                            </span>
                        </span>
                    </div>
                </CardTitle>
                <CardDescription>
                    Real-time performance metrics and cache statistics
                </CardDescription>
                <div className="mt-3 flex flex-wrap gap-2">
                    <InlineHint label="Coverage" detail="Portion of context matched to query; aim for >70%." />
                    <InlineHint label="Latency" detail="End-to-end time per query across planning, retrieval, generation." />
                    <InlineHint label="Cache health" detail="Embedding + query cache hit rates; clear if stale." />
                </div>
            </CardHeader>
            <CardContent className="space-y-6">
                {/* Quick Stats Grid */}
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
                    <StatCard
                        title="Total Queries"
                        value={totalQueries}
                        subtitle="All time"
                        icon={<Zap className="h-4 w-4" />}
                    />
                    <StatCard
                        title="Avg Latency"
                        value={`${avgLatency.toFixed(0)}ms`}
                        subtitle={avgLatency < 500 ? "Excellent" : avgLatency < 1000 ? "Good" : "Needs optimization"}
                        trend={avgLatency < 500 ? "up" : avgLatency < 1000 ? "neutral" : "down"}
                        icon={<Gauge className="h-4 w-4" />}
                    />
                    <StatCard
                        title="Avg Coverage"
                        value={`${(avgCoverage * 100).toFixed(0)}%`}
                        subtitle="Context relevance"
                        trend={avgCoverage > 0.7 ? "up" : avgCoverage > 0.5 ? "neutral" : "down"}
                        icon={<TrendingUp className="h-4 w-4" />}
                    />
                    <StatCard
                        title="Cache Hit Rate"
                        value={`${embeddingHitRate.toFixed(0)}%`}
                        subtitle={`${embeddingHits} hits / ${embeddingTotal} total`}
                        trend={embeddingHitRate > 50 ? "up" : "neutral"}
                        icon={<Database className="h-4 w-4" />}
                    />
                </div>

                {/* Recent Query Performance */}
                {recentQueries.length > 0 && (
                    <div className="space-y-3">
                        <h4 className="text-sm font-medium text-foreground">Recent Query Performance</h4>
                        <div className="space-y-3 rounded-lg border border-border/60 p-4">
                            {recentQueries.map((trace) => (
                                <PerformanceBar
                                    key={trace.id}
                                    label={trace.prompt.slice(0, 40) + (trace.prompt.length > 40 ? "..." : "")}
                                    value={getMetricNumber(trace.metrics.duration_ms as number | string)}
                                    max={2000}
                                />
                            ))}
                        </div>
                    </div>
                )}

                {/* Cache Details */}
                {cacheStats && (
                    <div className="space-y-4">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                            <h4 className="text-sm font-semibold text-foreground">Cache Controls</h4>
                            {onClearCache && (
                                <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={onClearCache}
                                    disabled={isClearingCache}
                                    className="text-xs"
                                >
                                    {isClearingCache ? "Clearing..." : "Clear Cache"}
                                </Button>
                            )}
                        </div>
                        <div className="grid gap-4 sm:grid-cols-2">
                            <div className="rounded-lg border border-border/60 bg-muted/20 p-4">
                                <h5 className="text-xs font-medium uppercase text-muted-foreground">Embedding Cache</h5>
                                <div className="mt-2 flex items-baseline gap-2">
                                    <span className="text-xl font-bold">{cacheStats.embeddings?.size || 0}</span>
                                    <span className="text-sm text-muted-foreground">items cached</span>
                                </div>
                                <p className="mt-1 text-xs text-muted-foreground">
                                    {embeddingHitRate.toFixed(1)}% hit rate
                                </p>
                            </div>
                            <div className="rounded-lg border border-border/60 bg-muted/20 p-4">
                                <h5 className="text-xs font-medium uppercase text-muted-foreground">Query Cache</h5>
                                <div className="mt-2 flex items-baseline gap-2">
                                    <span className="text-xl font-bold">{cacheStats.queries?.size || 0}</span>
                                    <span className="text-sm text-muted-foreground">items cached</span>
                                </div>
                                <p className="mt-1 text-xs text-muted-foreground">
                                    {queryHitRate.toFixed(1)}% hit rate
                                </p>
                            </div>
                        </div>
                    </div>
                )}

                {/* Empty State */}
                {totalQueries === 0 && (
                    <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border/70 bg-muted/10 p-12 text-center">
                        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted">
                            <Activity className="h-6 w-6 text-muted-foreground" />
                        </div>
                        <h3 className="mt-4 font-medium text-foreground">No metrics yet</h3>
                        <p className="mt-1 max-w-sm text-sm text-muted-foreground">
                            Run a query to start collecting performance data and analytics.
                        </p>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

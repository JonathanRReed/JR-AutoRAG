import { useMemo } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Activity, Database, Gauge, TrendingUp, Zap, Info, Clock, Layers } from "lucide-react";
import { StatCard, GaugeRing, Sparkline, DurationBar } from "./LiveMetricsChart";
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
            <Info className="h-3.5 w-3.5 text-muted-foreground" />
            <span className="truncate">{label}</span>
        </span>
    );
}

const getMetricNumber = (value: number | string | undefined) =>
    typeof value === "number" && Number.isFinite(value) ? value : 0;

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

    // Recent queries for latency sparkline
    const recentQueries = traces.slice(-10);
    const latencySparkline = useMemo(() =>
        recentQueries.map(t => getMetricNumber(t.metrics.duration_ms as number | string)),
        [recentQueries]
    );
    const coverageSparkline = useMemo(() =>
        recentQueries.map(t => getMetricNumber(t.metrics.coverage as number | string) * 100),
        [recentQueries]
    );

    // Recent 5 for detail view
    const displayQueries = traces.slice(-5).reverse();

    return (
        <Card className="overflow-hidden">
            <CardHeader className="glass-panel">
                <CardTitle className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                        <Activity className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                        <span className="flex items-center gap-2">
                            System Metrics
                            <span className="live-indicator">
                                <span className="live-dot" />
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
            <CardContent className="space-y-6 p-6">
                {/* Quick Stats Grid with Premium Cards */}
                <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 stagger-in">
                    <StatCard
                        title="Total Queries"
                        value={totalQueries}
                        subtitle="All time queries"
                        icon={<Zap className="h-5 w-5" />}
                        sparklineData={latencySparkline.slice(-6)}
                    />
                    <StatCard
                        title="Avg Latency"
                        value={`${avgLatency.toFixed(0)}ms`}
                        subtitle={avgLatency < 500 ? "Excellent" : avgLatency < 1000 ? "Good" : "Needs optimization"}
                        icon={<Clock className="h-5 w-5" />}
                        sparklineData={latencySparkline}
                    />
                    <StatCard
                        title="Avg Coverage"
                        value={`${(avgCoverage * 100).toFixed(0)}%`}
                        subtitle="Context relevance"
                        gauge={avgCoverage * 100}
                        icon={<TrendingUp className="h-5 w-5" />}
                    />
                    <StatCard
                        title="Cache Hit Rate"
                        value={`${embeddingHitRate.toFixed(0)}%`}
                        subtitle={`${embeddingHits} hits / ${embeddingTotal} total`}
                        gauge={embeddingHitRate}
                        icon={<Database className="h-5 w-5" />}
                    />
                </div>

                {/* Recent Query Performance with Duration Bars */}
                {displayQueries.length > 0 && (
                    <div className="space-y-3">
                        <div className="flex items-center justify-between">
                            <h4 className="text-sm font-semibold text-foreground flex items-center gap-2">
                                <Gauge className="h-4 w-4 text-muted-foreground" />
                                Recent Query Performance
                            </h4>
                            <span className="text-xs text-muted-foreground">
                                Last {displayQueries.length} queries
                            </span>
                        </div>
                        <div className="glass-card rounded-lg p-4 space-y-3">
                            {displayQueries.map((trace) => (
                                <DurationBar
                                    key={trace.id}
                                    label={trace.prompt.slice(0, 45) + (trace.prompt.length > 45 ? "..." : "")}
                                    value={getMetricNumber(trace.metrics.duration_ms as number | string)}
                                    max={2000}
                                />
                            ))}
                        </div>
                    </div>
                )}

                {/* Cache Details with Enhanced Visuals */}
                {cacheStats && (
                    <div className="space-y-4">
                        <div className="flex flex-wrap items-center justify-between gap-3">
                            <h4 className="text-sm font-semibold text-foreground flex items-center gap-2">
                                <Layers className="h-4 w-4 text-muted-foreground" />
                                Cache Controls
                            </h4>
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
                            <div className="glass-card rounded-lg p-4">
                                <div className="flex items-start justify-between">
                                    <div>
                                        <h5 className="text-xs font-medium uppercase text-muted-foreground">Embedding Cache</h5>
                                        <div className="mt-2 flex items-baseline gap-2">
                                            <span className="text-2xl font-bold stat-value">{cacheStats.embeddings?.size || 0}</span>
                                            <span className="text-sm text-muted-foreground">items cached</span>
                                        </div>
                                        <p className="mt-1 text-xs text-muted-foreground">
                                            {embeddingHitRate.toFixed(1)}% hit rate
                                        </p>
                                    </div>
                                    <GaugeRing value={embeddingHitRate} size={44} />
                                </div>
                            </div>
                            <div className="glass-card rounded-lg p-4">
                                <div className="flex items-start justify-between">
                                    <div>
                                        <h5 className="text-xs font-medium uppercase text-muted-foreground">Query Cache</h5>
                                        <div className="mt-2 flex items-baseline gap-2">
                                            <span className="text-2xl font-bold stat-value">{cacheStats.queries?.size || 0}</span>
                                            <span className="text-sm text-muted-foreground">items cached</span>
                                        </div>
                                        <p className="mt-1 text-xs text-muted-foreground">
                                            {queryHitRate.toFixed(1)}% hit rate
                                        </p>
                                    </div>
                                    <GaugeRing value={queryHitRate} size={44} />
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Empty State */}
                {totalQueries === 0 && (
                    <div className="flex flex-col items-center justify-center glass-card rounded-lg p-12 text-center fade-in">
                        <div className="flex h-14 w-14 items-center justify-center rounded-full bg-muted/50">
                            <Activity className="h-7 w-7 text-muted-foreground" />
                        </div>
                        <h3 className="mt-4 font-semibold text-foreground">No metrics yet</h3>
                        <p className="mt-2 max-w-sm text-sm text-muted-foreground">
                            Run a query to start collecting performance data and analytics.
                        </p>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

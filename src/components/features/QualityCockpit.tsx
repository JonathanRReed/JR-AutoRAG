import { useEffect, useMemo, useState } from "react";
import { Activity, ArrowUpRight, Beaker, Download, FileCheck2, FileSearch, RefreshCw, ShieldCheck, Trophy } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { EmptyState, ProgressBar, StatusBadge, StatCard } from "@/components/ui/shared";
import type {
  DocumentOut,
  DocumentPreview,
  EvalRunSummary,
  ExperimentRun,
  QualityRecommendations,
} from "@/types";

type QualityCockpitProps = {
  documents: DocumentOut[];
  buildUrl: (path: string) => string;
  buildHeaders: (extra?: HeadersInit) => Headers;
  onPresetPromoted: () => void;
};

const toMessage = (error: unknown) => (error instanceof Error ? error.message : String(error));

const CLIENT_READINESS_REQUIRED_TAGS = [
  "client-readiness",
  "mixed-format",
  "prompt-injection",
  "abstention",
  "binary-retrieval",
  "agentic-retrieval",
] as const;

const CLIENT_READINESS_METRIC_THRESHOLDS = {
  "Recall": 0.7,
  "Citation": 0.85,
  "Faithfulness": 0.9,
  "Completeness": 0.7,
} as const;

const readJson = async <T,>(response: Response): Promise<T> => {
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json() as Promise<T>;
};

const formatScore = (value?: number) =>
  typeof value === "number" && Number.isFinite(value) ? `${Math.round(value * 100)}%` : "0%";

const shortHash = (value?: string) => (value ? value.slice(0, 12) : "No artifact");

const asRecord = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const auditString = (run: EvalRunSummary | undefined, section: string, key: string) => {
  const item = asRecord(run?.audit?.[section])[key];
  return typeof item === "string" ? item : "";
};

export function QualityCockpit({ documents, buildUrl, buildHeaders, onPresetPromoted }: QualityCockpitProps) {
  const [recommendations, setRecommendations] = useState<QualityRecommendations | null>(null);
  const [experiments, setExperiments] = useState<ExperimentRun[]>([]);
  const [evalRuns, setEvalRuns] = useState<EvalRunSummary[]>([]);
  const [preview, setPreview] = useState<DocumentPreview | null>(null);
  const [selectedDocumentId, setSelectedDocumentId] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isRunningExperiment, setIsRunningExperiment] = useState(false);
  const [isRunningClientReadiness, setIsRunningClientReadiness] = useState(false);
  const [downloadingReportId, setDownloadingReportId] = useState("");
  const [error, setError] = useState("");

  const selectedDocument = useMemo(
    () => documents.find((doc) => doc.id === selectedDocumentId) ?? documents[0],
    [documents, selectedDocumentId],
  );

  const refresh = async () => {
    setIsLoading(true);
    setError("");
    try {
      const [recommendationData, experimentData, evalData] = await Promise.all([
        fetch(buildUrl("/config/recommendations"), { headers: buildHeaders() }).then((response) =>
          readJson<QualityRecommendations>(response),
        ),
        fetch(buildUrl("/experiments"), { headers: buildHeaders() }).then((response) =>
          readJson<ExperimentRun[]>(response),
        ),
        fetch(buildUrl("/evaluation/runs"), { headers: buildHeaders() }).then((response) =>
          readJson<EvalRunSummary[]>(response),
        ),
      ]);
      setRecommendations(recommendationData);
      setExperiments(experimentData);
      setEvalRuns(evalData);
    } catch (caught) {
      setError(toMessage(caught));
    } finally {
      setIsLoading(false);
    }
  };

  const loadPreview = async (documentId: string) => {
    if (!documentId) {
      setPreview(null);
      return;
    }
    setError("");
    try {
      const data = await fetch(buildUrl(`/documents/${documentId}/preview`), { headers: buildHeaders() }).then(
        (response) => readJson<DocumentPreview>(response),
      );
      setPreview(data);
      setSelectedDocumentId(documentId);
    } catch (caught) {
      setError(toMessage(caught));
    }
  };

  const runExperiment = async () => {
    setIsRunningExperiment(true);
    setError("");
    try {
      const run = await fetch(buildUrl("/experiments"), {
        method: "POST",
        headers: buildHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          name: "Local quality advisory",
          description: "Parser, retrieval, OCR, graph, and RAPTOR advisory sweep",
          questions: ["What is this corpus about?", "Which sources support the answer?"],
        }),
      }).then((response) => readJson<ExperimentRun>(response));
      setExperiments((current) => [run, ...current.filter((item) => item.id !== run.id)]);
      await refresh();
    } catch (caught) {
      setError(toMessage(caught));
    } finally {
      setIsRunningExperiment(false);
    }
  };

  const runClientReadinessBenchmark = async () => {
    setIsRunningClientReadiness(true);
    setError("");
    try {
      await fetch(buildUrl("/evaluation/golden-sets/builtins"), {
        method: "POST",
        headers: buildHeaders(),
      }).then((response) => readJson<unknown>(response));
      const run = await fetch(buildUrl("/evaluation/batch/client_readiness"), {
        method: "POST",
        headers: buildHeaders(),
      }).then((response) => readJson<EvalRunSummary>(response));
      setEvalRuns((current) => [run, ...current.filter((item) => item.run_id !== run.run_id)]);
      await refresh();
    } catch (caught) {
      setError(toMessage(caught));
    } finally {
      setIsRunningClientReadiness(false);
    }
  };

  const promoteExperiment = async (runId: string) => {
    setError("");
    try {
      await fetch(buildUrl(`/experiments/${runId}/promote`), {
        method: "POST",
        headers: buildHeaders(),
      }).then((response) => readJson<unknown>(response));
      await refresh();
      onPresetPromoted();
    } catch (caught) {
      setError(toMessage(caught));
    }
  };

  const downloadEvalReport = async (run: EvalRunSummary) => {
    setDownloadingReportId(run.run_id);
    setError("");
    try {
      const report = await fetch(buildUrl(`/evaluation/runs/${run.run_id}/report`), {
        headers: buildHeaders(),
      }).then((response) => readJson<Record<string, unknown>>(response));
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `jr-autorag-eval-${run.run_id}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (caught) {
      setError(toMessage(caught));
    } finally {
      setDownloadingReportId("");
    }
  };

  useEffect(() => {
    void refresh();
  }, []);

  useEffect(() => {
    void refresh();
  }, [documents.length]);

  useEffect(() => {
    if (!selectedDocumentId && documents[0]?.id) {
      setSelectedDocumentId(documents[0].id);
    }
  }, [documents, selectedDocumentId]);

  const latestExperiment = experiments[0];
  const faithfulness = latestExperiment?.metrics.find((metric) => metric.name === "faithfulness")?.value ?? 0;
  const contextPrecision = latestExperiment?.metrics.find((metric) => metric.name === "context_precision")?.value ?? 0;
  const latestEval = evalRuns[0];
  const clientReadinessEval = evalRuns.find((run) => run.golden_set_name === "client_readiness");
  const clientReadinessTagCounts = useMemo(() => {
    const goldenSet = asRecord(asRecord(clientReadinessEval?.audit).golden_set);
    return asRecord(goldenSet.tag_counts);
  }, [clientReadinessEval]);
  const missingClientReadinessTags = CLIENT_READINESS_REQUIRED_TAGS.filter(
    (tag) => Number(clientReadinessTagCounts[tag] ?? 0) <= 0,
  );
  const clientReadinessMetrics = {
    "Recall": clientReadinessEval?.retrieval_metrics.recall_at_k ?? 0,
    "Citation": clientReadinessEval?.retrieval_metrics.citation_coverage ?? 0,
    "Faithfulness": clientReadinessEval?.answer_metrics.faithfulness ?? 0,
    "Completeness": clientReadinessEval?.answer_metrics.completeness ?? 0,
  };
  const failedClientReadinessMetrics = Object.entries(CLIENT_READINESS_METRIC_THRESHOLDS).filter(
    ([name, threshold]) => clientReadinessMetrics[name as keyof typeof CLIENT_READINESS_METRIC_THRESHOLDS] < threshold,
  );
  const clientReadinessReady =
    Boolean(clientReadinessEval?.report_sha256) &&
    missingClientReadinessTags.length === 0 &&
    failedClientReadinessMetrics.length === 0;
  const latestCorpusFingerprint = auditString(latestEval, "corpus", "fingerprint");
  const latestGoldenFingerprint = auditString(latestEval, "golden_set", "fingerprint");
  const corpusDocumentCount = Math.max(documents.length, recommendations?.document_count ?? 0);
  const lowConfidenceCount = recommendations?.low_confidence_documents ?? 0;

  return (
    <section className="flex flex-col gap-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold tracking-tight">Quality Cockpit</h2>
          <p className="text-sm text-muted-foreground">
            Corpus health, extraction previews, eval runs, and local preset promotion.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Button variant="outline" size="sm" onClick={refresh} disabled={isLoading}>
            <RefreshCw data-icon="inline-start" />
            {isLoading ? "Refreshing" : "Refresh"}
          </Button>
          <Button size="sm" onClick={runExperiment} disabled={isRunningExperiment}>
            <Beaker data-icon="inline-start" />
            {isRunningExperiment ? "Running" : "Run Advisory"}
          </Button>
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
          {error}
        </div>
      )}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard
          title="Corpus"
          value={corpusDocumentCount}
          subtitle={`${lowConfidenceCount} low confidence`}
          icon={<FileSearch />}
          trend={lowConfidenceCount > 0 ? "down" : "neutral"}
        />
        <StatCard
          title="Faithfulness"
          value={formatScore(faithfulness)}
          subtitle={latestExperiment?.winning_preset ? `Preset ${latestExperiment.winning_preset}` : "No matrix run yet"}
          icon={<ShieldCheck />}
          trend={faithfulness > 0.75 ? "up" : "neutral"}
        />
        <StatCard
          title="Context Precision"
          value={formatScore(contextPrecision)}
          subtitle={`${experiments.length} experiment run(s)`}
          icon={<Activity />}
          trend={contextPrecision > 0.75 ? "up" : "neutral"}
        />
        <StatCard
          title="Eval Runs"
          value={evalRuns.length}
          subtitle={latestEval?.report_sha256 ? `Report ${shortHash(latestEval.report_sha256)}` : latestEval ? latestEval.golden_set_name : "No golden run yet"}
          icon={<Trophy />}
          trend={evalRuns.length ? "up" : "neutral"}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.05fr),minmax(380px,0.95fr)]">
        <Card>
          <CardHeader>
            <CardTitle>Extraction Preview</CardTitle>
            <CardDescription>Inspect parser confidence, headings, tables, OCR metadata, and chunk provenance hints.</CardDescription>
            <CardAction>
              <Button
                variant="outline"
                size="sm"
                disabled={!selectedDocument}
                onClick={() => selectedDocument && void loadPreview(selectedDocument.id)}
              >
                <FileSearch data-icon="inline-start" />
                Preview
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {documents.length === 0 ? (
              <EmptyState icon={<FileSearch />} title="No corpus yet" description="Ingest documents to inspect parser quality." />
            ) : (
              <>
                <label className="flex flex-col gap-2 text-sm">
                  <span className="font-medium text-foreground">Document</span>
                  <select
                    className="h-9 rounded-md border border-border bg-background px-3 text-sm"
                    value={selectedDocument?.id ?? ""}
                    onChange={(event) => {
                      setSelectedDocumentId(event.target.value);
                      void loadPreview(event.target.value);
                    }}
                  >
                    {documents.map((doc) => (
                      <option key={doc.id} value={doc.id}>
                        {doc.title}
                      </option>
                    ))}
                  </select>
                </label>
                {preview ? (
                  <div className="flex flex-col gap-4">
                    <div className="grid gap-3 sm:grid-cols-3">
                      <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
                        <div className="text-xs text-muted-foreground">Parser</div>
                        <div className="mt-1 text-sm font-medium">{preview.parser_provider}</div>
                      </div>
                      <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
                        <div className="text-xs text-muted-foreground">Pages</div>
                        <div className="mt-1 text-sm font-medium">{preview.page_count}</div>
                      </div>
                      <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
                        <div className="text-xs text-muted-foreground">Confidence</div>
                        <div className="mt-2">
                          <ProgressBar value={preview.confidence * 100} showValue size="sm" />
                        </div>
                      </div>
                    </div>
                    <div className="max-h-[360px] overflow-auto rounded-lg border border-border/60">
                      {preview.blocks.slice(0, 16).map((block, index) => (
                        <div key={`${block.type}-${index}`} className="border-b border-border/40 p-3 last:border-b-0">
                          <div className="mb-2 flex flex-wrap items-center gap-2">
                            <StatusBadge status={block.type === "heading" ? "info" : block.type === "table" ? "warning" : "neutral"}>
                              {block.type}
                            </StatusBadge>
                            <span className="text-xs text-muted-foreground">Page {block.page ?? 1}</span>
                          </div>
                          <p className="text-sm leading-relaxed text-foreground">{block.text || "No extracted text"}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <EmptyState icon={<FileSearch />} title="Preview not loaded" description="Choose a document and load its extraction preview." />
                )}
              </>
            )}
          </CardContent>
        </Card>

        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Evaluation Evidence</CardTitle>
              <CardDescription>Golden-run artifacts, corpus fingerprints, and exportable quality receipts.</CardDescription>
              <CardAction>
                <div className="flex flex-wrap items-center justify-end gap-2">
                  <Button
                    variant="secondary"
                    size="sm"
                    disabled={isRunningClientReadiness}
                    onClick={() => void runClientReadinessBenchmark()}
                  >
                    <ShieldCheck data-icon="inline-start" />
                    {isRunningClientReadiness ? "Running" : "Run Client Readiness"}
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    disabled={!latestEval?.report_sha256 || downloadingReportId === latestEval?.run_id}
                    onClick={() => latestEval && void downloadEvalReport(latestEval)}
                  >
                    <Download data-icon="inline-start" />
                    {downloadingReportId === latestEval?.run_id ? "Exporting" : "Export"}
                  </Button>
                </div>
              </CardAction>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              <div className="rounded-lg border border-border/60 bg-background p-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div className="min-w-0">
                    <div className="text-sm font-medium">Client Readiness</div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {clientReadinessEval?.report_sha256
                        ? `Report ${shortHash(clientReadinessEval.report_sha256)}`
                        : "No client handoff receipt yet"}
                    </div>
                  </div>
                  <StatusBadge status={clientReadinessReady ? "success" : clientReadinessEval ? "warning" : "neutral"}>
                    {clientReadinessReady ? "ready" : clientReadinessEval ? "needs rerun" : "not run"}
                  </StatusBadge>
                </div>
                {clientReadinessEval ? (
                  <div className="mt-3 grid gap-2 sm:grid-cols-2">
                    {Object.entries(clientReadinessMetrics).map(([name, value]) => (
                      <div key={name} className="flex items-center justify-between gap-3 rounded-md border border-border/50 px-3 py-2">
                        <span className="text-xs text-muted-foreground">{name}</span>
                        <span className="font-mono text-xs font-semibold">{formatScore(value)}</span>
                      </div>
                    ))}
                  </div>
                ) : null}
                {!clientReadinessReady ? (
                  <div className="mt-3 text-xs text-muted-foreground">
                    {missingClientReadinessTags.length > 0
                      ? `Missing coverage: ${missingClientReadinessTags.join(", ")}`
                      : failedClientReadinessMetrics.length > 0
                        ? `Below gate: ${failedClientReadinessMetrics.map(([name]) => name).join(", ")}`
                        : "Run the benchmark after ingesting representative client documents."}
                  </div>
                ) : null}
              </div>
              {!latestEval ? (
                <EmptyState icon={<FileCheck2 />} title="No golden reports" description="Run a golden evaluation to create a report artifact." />
              ) : (
                <>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
                      <div className="text-xs text-muted-foreground">Golden Set</div>
                      <div className="mt-1 truncate text-sm font-medium">{latestEval.golden_set_name}</div>
                    </div>
                    <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
                      <div className="text-xs text-muted-foreground">Report SHA</div>
                      <div className="mt-1 font-mono text-sm font-medium">{shortHash(latestEval.report_sha256)}</div>
                    </div>
                    <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
                      <div className="text-xs text-muted-foreground">Corpus</div>
                      <div className="mt-1 font-mono text-sm font-medium">{shortHash(latestCorpusFingerprint)}</div>
                    </div>
                    <div className="rounded-lg border border-border/60 bg-muted/20 p-3">
                      <div className="text-xs text-muted-foreground">Golden Fingerprint</div>
                      <div className="mt-1 font-mono text-sm font-medium">{shortHash(latestGoldenFingerprint)}</div>
                    </div>
                  </div>
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="flex items-center justify-between gap-3 rounded-lg border border-border/60 bg-background p-3">
                      <span className="text-sm text-muted-foreground">Recall</span>
                      <span className="font-mono text-sm font-semibold">{formatScore(latestEval.retrieval_metrics.recall_at_k)}</span>
                    </div>
                    <div className="flex items-center justify-between gap-3 rounded-lg border border-border/60 bg-background p-3">
                      <span className="text-sm text-muted-foreground">Faithfulness</span>
                      <span className="font-mono text-sm font-semibold">{formatScore(latestEval.answer_metrics.faithfulness)}</span>
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recommendations</CardTitle>
              <CardDescription>Local-first actions ranked from the current corpus and config.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              {(recommendations?.recommendations.length ?? 0) === 0 ? (
                <EmptyState icon={<ShieldCheck />} title="No recommendations" description="Current quality signals are clean." />
              ) : (
                recommendations?.recommendations.map((item) => (
                  <div key={item.id} className="rounded-lg border border-border/60 bg-background p-3">
                    <div className="mb-2 flex items-center justify-between gap-3">
                      <div className="font-medium text-sm">{item.title}</div>
                      <StatusBadge status={item.priority === "high" ? "error" : item.priority === "medium" ? "warning" : "neutral"}>
                        {item.priority}
                      </StatusBadge>
                    </div>
                    <p className="text-sm leading-relaxed text-muted-foreground">{item.detail}</p>
                  </div>
                ))
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Experiment Leaderboard</CardTitle>
              <CardDescription>Promote the latest winning preset after reviewing its metric set.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              {experiments.length === 0 ? (
                <EmptyState icon={<Beaker />} title="No experiment runs" description="Run a local quality advisory to create the first leaderboard entry." />
              ) : (
                experiments.slice(0, 6).map((run) => {
                  const score = run.metrics.find((metric) => metric.name === "faithfulness")?.value ?? 0;
                  return (
                    <div key={run.id} className="rounded-lg border border-border/60 bg-background p-3">
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0">
                          <div className="truncate text-sm font-medium">{run.config.name}</div>
                          <div className="text-xs text-muted-foreground">{run.created_at}</div>
                        </div>
                        <Button variant="outline" size="sm" onClick={() => void promoteExperiment(run.id)}>
                          <ArrowUpRight data-icon="inline-start" />
                          Promote
                        </Button>
                      </div>
                      <div className="mt-3 flex items-center gap-3">
                        <div className="min-w-24 text-sm font-medium">{formatScore(score)}</div>
                        <div className="flex-1">
                          <ProgressBar value={score * 100} showValue={false} size="sm" />
                        </div>
                        <StatusBadge status={run.promoted_at ? "success" : "neutral"}>
                          {run.promoted_at ? "promoted" : run.winning_preset ?? "balanced"}
                        </StatusBadge>
                      </div>
                    </div>
                  );
                })
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </section>
  );
}

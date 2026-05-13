import { ArrowRight, CheckCircle2, Database, FileSearch, MessageSquare, Network, Play, ShieldCheck } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Empty, EmptyContent, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from "@/components/ui/empty";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import type { OnboardingExampleQuery, OnboardingState } from "@/types";

type OnboardingFlowProps = {
  onboarding: OnboardingState | null;
  apiReady: boolean;
  docsReady: boolean;
  modelsReady: boolean;
  isSeedingDemo: boolean;
  activeMode: "guided" | "advanced";
  onModeChange: (mode: "guided" | "advanced") => void;
  onSeedDemo: () => void;
  onOpenTab: (tab: "config" | "documents" | "query" | "quality" | "metrics") => void;
  onAskExample: (query: string) => void;
};

const steps = [
  { id: "connect", title: "Connect", description: "API health and local services", icon: Network },
  { id: "knowledge", title: "Knowledge", description: "Client documents", icon: Database },
  { id: "ask", title: "First Answer", description: "Streaming answer with citations", icon: MessageSquare },
  { id: "inspect", title: "Inspect", description: "Trace, sources, and quality", icon: FileSearch },
];

const fallbackQueries: OnboardingExampleQuery[] = [
  {
    query: "What should an operator verify before a client install?",
    category: "install",
    expected_docs: ["JR AutoRAG Evaluation Brief"],
  },
  {
    query: "Which RAG controls should be enabled for a high-risk knowledge base?",
    category: "research",
    expected_docs: ["State of the Art RAG Playbook"],
  },
  {
    query: "How should a client handoff explain evidence, traces, and quality receipts?",
    category: "workflow",
    expected_docs: ["Project Manager Demo Scenario"],
  },
  {
    query: "Compare hybrid search and RAPTOR for enterprise documents",
    category: "comparison",
    expected_docs: ["Advanced Retrieval Techniques"],
  },
];

function readinessProgress(apiReady: boolean, docsReady: boolean, modelsReady: boolean) {
  const checks = [apiReady, docsReady, modelsReady];
  return Math.round((checks.filter(Boolean).length / checks.length) * 100);
}

function ExampleQueryButton({
  query,
  onAskExample,
}: {
  query: OnboardingExampleQuery;
  onAskExample: (query: string) => void;
}) {
  return (
    <button
      type="button"
      onClick={() => onAskExample(query.query)}
      className="group flex min-h-24 flex-col items-start justify-between gap-3 rounded-lg border border-border/60 bg-card p-4 text-left transition-colors hover:border-primary/40 hover:bg-muted/20"
    >
      <div className="flex w-full items-center justify-between gap-3">
        <Badge variant="outline">{query.category}</Badge>
        <ArrowRight data-icon="inline-end" className="opacity-0 transition-opacity group-hover:opacity-100" />
      </div>
      <span className="text-sm font-medium leading-relaxed text-foreground">{query.query}</span>
    </button>
  );
}

export function OnboardingFlow({
  onboarding,
  apiReady,
  docsReady,
  modelsReady,
  isSeedingDemo,
  activeMode,
  onModeChange,
  onSeedDemo,
  onOpenTab,
  onAskExample,
}: OnboardingFlowProps) {
  const onboardingState = onboarding ?? {
    flow: {
      steps: [],
      current_step: 0,
      progress: 0,
      is_complete: false,
    },
    demo_mode: false,
    demo_seeded: docsReady,
    document_count: docsReady ? 1 : 0,
    demo_document_count: docsReady ? 1 : 0,
    sample_documents: [],
    example_queries: fallbackQueries,
  };
  const progress = readinessProgress(apiReady, docsReady, modelsReady);
  const demoSeeded = Boolean(onboardingState.demo_seeded || onboardingState.demo_document_count);
  const examples = onboardingState.example_queries.slice(0, 4);

  return (
    <Card className="overflow-hidden">
      <CardHeader>
        <CardTitle>Install Onboarding</CardTitle>
        <CardDescription>Connect services, ingest client knowledge, then prove answer quality with evidence.</CardDescription>
        <CardAction>
          <ToggleGroup aria-label="Onboarding mode">
            <ToggleGroupItem pressed={activeMode === "guided"} onClick={() => onModeChange("guided")}>
              Guided
            </ToggleGroupItem>
            <ToggleGroupItem pressed={activeMode === "advanced"} onClick={() => onModeChange("advanced")}>
              Advanced
            </ToggleGroupItem>
          </ToggleGroup>
        </CardAction>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-4">
            <Alert variant={onboardingState.demo_mode ? "info" : "default"}>
              <ShieldCheck data-icon="inline-start" />
              <AlertTitle>{onboardingState.demo_mode ? "Demo Walkthrough Mode" : "Client Install Mode"}</AlertTitle>
              <AlertDescription>
                {onboardingState.demo_mode
                  ? "This run uses a temporary local directory for sales or evaluation walkthroughs."
                  : "Persistent local storage is active. Client installs start with the customer's documents and finish with evidence receipts."}
              </AlertDescription>
            </Alert>

            <div className="grid gap-3 md:grid-cols-4">
              {steps.map((step) => {
                const Icon = step.icon;
                const complete =
                  (step.id === "connect" && apiReady) ||
                  (step.id === "knowledge" && (docsReady || demoSeeded)) ||
                  (step.id === "ask" && docsReady) ||
                  (step.id === "inspect" && docsReady);
                return (
                  <div key={step.id} className="rounded-lg border border-border/60 bg-muted/10 p-3">
                    <div className="flex items-center justify-between gap-2">
                      <Icon className="text-muted-foreground" />
                      {complete ? <CheckCircle2 className="text-primary" /> : <span className="size-2 rounded-full bg-muted-foreground/40" />}
                    </div>
                    <div className="mt-3 text-sm font-medium text-foreground">{step.title}</div>
                    <div className="mt-1 text-xs leading-relaxed text-muted-foreground">{step.description}</div>
                  </div>
                );
              })}
            </div>

            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between gap-3 text-xs text-muted-foreground">
                <span>Readiness</span>
                <span>{progress}%</span>
              </div>
              <Progress value={progress} />
            </div>
          </div>

          <details className="rounded-lg border border-border/60 bg-muted/10 p-3">
            <summary className="cursor-pointer text-sm font-medium text-muted-foreground">Demo walkthrough options</summary>
            <div className="mt-3 flex flex-col gap-3">
              <div className="text-xs leading-relaxed text-muted-foreground">
                Load sample documents only for sales or evaluator walkthroughs. Client installs should use the Documents tab.
              </div>
              <Separator />
              <div className="grid grid-cols-3 gap-2 text-xs">
                <Badge variant={apiReady ? "default" : "muted"}>API</Badge>
                <Badge variant={modelsReady ? "default" : "muted"}>Models</Badge>
                <Badge variant={docsReady ? "default" : "muted"}>Corpus</Badge>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant="outline" size="sm" onClick={onSeedDemo} disabled={!apiReady || isSeedingDemo}>
                  <Play data-icon="inline-start" />
                  {isSeedingDemo ? "Loading Demo" : demoSeeded ? "Refresh Demo" : "Load Demo"}
                </Button>
                <Button variant="ghost" size="sm" onClick={() => onOpenTab("documents")}>
                  Documents
                </Button>
              </div>
            </div>
          </details>
        </div>

        {activeMode === "guided" ? (
          <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
            {examples.map((query) => (
              <ExampleQueryButton key={query.query} query={query} onAskExample={onAskExample} />
            ))}
          </div>
        ) : (
          <Empty className="min-h-[160px]">
            <EmptyHeader>
              <EmptyMedia>
                <FileSearch />
              </EmptyMedia>
              <EmptyTitle>Advanced Path</EmptyTitle>
              <EmptyDescription>
                Configure providers, tune retrieval, run advisory experiments, and inspect parser previews before client handoff.
              </EmptyDescription>
            </EmptyHeader>
            <EmptyContent>
              <Button variant="outline" onClick={() => onOpenTab("config")}>Open Configuration</Button>
              <Button variant="outline" onClick={() => onOpenTab("quality")}>Open Quality Cockpit</Button>
            </EmptyContent>
          </Empty>
        )}
      </CardContent>
    </Card>
  );
}

export default OnboardingFlow;

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import type { QueryResponse } from "@/types";

interface ChatInterfaceProps {
    question: string;
    setQuestion: (value: string) => void;
    isQuerying: boolean;
    handleAsk: () => void;
    queryResult: QueryResponse | null;
}

export function ChatInterface({
    question,
    setQuestion,
    isQuerying,
    handleAsk,
    queryResult,
}: ChatInterfaceProps) {
    return (
        <Card>
            <CardHeader>
                <CardTitle>Ask a Question</CardTitle>
                <CardDescription>Send a query through the full RAG stack.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div className="flex flex-col gap-2 sm:flex-row">
                    <Input value={question} onChange={e => setQuestion(e.target.value)} />
                    <Button disabled={isQuerying} onClick={handleAsk}>
                        {isQuerying ? "Querying..." : "Run Query"}
                    </Button>
                </div>
                {queryResult && (
                    <div className="space-y-4 rounded-lg border border-border p-4">
                        <div>
                            <p className="text-sm font-semibold text-muted-foreground">Answer</p>
                            <p className="text-base text-foreground">{queryResult.answer}</p>
                        </div>
                        <div>
                            <p className="text-sm font-semibold text-muted-foreground">Evidence ({queryResult.chunks.length})</p>
                            <ul className="mt-2 space-y-1 text-sm text-muted-foreground">
                                {queryResult.chunks.map(chunk => (
                                    <li key={chunk.id}>
                                        <span className="font-medium text-foreground">{chunk.title}</span> · score {chunk.score.toFixed(3)} ·
                                        {" "}
                                        {chunk.snippet.slice(0, 120)}
                                    </li>
                                ))}
                            </ul>
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

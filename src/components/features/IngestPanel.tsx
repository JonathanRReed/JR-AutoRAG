import { type ChangeEvent, type DragEvent, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import {
    Calendar,
    CheckCircle2,
    FileText,
    FolderOpen,
    HardDrive,
    Inbox,
    Loader2,
    Trash2,
    Upload,
    XCircle,
    ShieldCheck,
    Info,
} from "lucide-react";
import type { DocumentOut, IngestResponse } from "@/types";

interface UploadProgress {
    filename: string;
    status: "pending" | "uploading" | "processing" | "done" | "error";
    error?: string;
}

interface IngestPanelProps {
    ingestTitle: string;
    setIngestTitle: (value: string) => void;
    ingestText: string;
    setIngestText: (value: string) => void;
    isIngesting: boolean;
    handleIngest: () => void;
    ingestSync: boolean;
    setIngestSync: (value: boolean) => void;
    documents: DocumentOut[];
    handleDeleteDocument: (id: string, title: string) => void;
    handleDeleteAllDocuments: () => void;
    isUploadingFile: boolean;
    uploadFile: (file: File) => Promise<IngestResponse>;
    waitForDocumentReady: (documentId: string, label?: string) => Promise<void>;
    fileInputRef: React.RefObject<HTMLInputElement | null>;
    formatDateTime: (value?: string) => string;
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

export function IngestPanel({
    ingestTitle,
    setIngestTitle,
    ingestText,
    setIngestText,
    isIngesting,
    handleIngest,
    ingestSync,
    setIngestSync,
    documents,
    handleDeleteDocument,
    handleDeleteAllDocuments,
    isUploadingFile,
    uploadFile,
    waitForDocumentReady,
    fileInputRef,
    formatDateTime,
}: IngestPanelProps) {
    const [uploadQueue, setUploadQueue] = useState<UploadProgress[]>([]);
    const [isDragOver, setIsDragOver] = useState(false);

    const processFiles = async (files: FileList | File[]) => {
        const fileArray = Array.from(files);
        const validFiles = fileArray.filter(f =>
            /\.(pdf|docx?|txt|md|markdown)$/i.test(f.name)
        );

        if (validFiles.length === 0) {
            return;
        }

        const initialQueue: UploadProgress[] = validFiles.map(f => ({
            filename: f.name,
            status: "pending",
        }));
        setUploadQueue(initialQueue);

        for (let i = 0; i < validFiles.length; i++) {
            const file = validFiles[i];
            if (!file) continue;

            setUploadQueue(prev => prev.map((item, idx) =>
                idx === i ? { ...item, status: "uploading" } : item
            ));

            try {
                const result = await uploadFile(file);
                setUploadQueue(prev => prev.map((item, idx) =>
                    idx === i ? { ...item, status: "processing" } : item
                ));
                await waitForDocumentReady(result.document_id, file.name);
                setUploadQueue(prev => prev.map((item, idx) =>
                    idx === i ? { ...item, status: "done" } : item
                ));
            } catch (err) {
                setUploadQueue(prev => prev.map((item, idx) =>
                    idx === i ? { ...item, status: "error", error: String(err) } : item
                ));
            }
        }

        setTimeout(() => setUploadQueue([]), 5000);
    };

    const handleFileInputChange = (event: ChangeEvent<HTMLInputElement>) => {
        const files = event.target.files;
        if (files && files.length > 0) {
            void processFiles(files);
        }
    };

    const handleDrop = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setIsDragOver(false);
        const files = event.dataTransfer.files;
        if (files.length > 0) {
            void processFiles(files);
        }
    };

    const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setIsDragOver(true);
    };

    const handleDragLeave = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        setIsDragOver(false);
    };

    return (
        <Card className="overflow-hidden">
            <CardHeader className="bg-muted/20">
                <CardTitle className="flex items-center gap-3">
                    <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
                        <FileText className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                        <span>Document Management</span>
                        {documents.length > 0 && (
                            <span className="ml-2 inline-flex h-5 items-center justify-center rounded-full bg-muted px-2 text-[10px] font-semibold text-muted-foreground">
                                {documents.length}
                            </span>
                        )}
                    </div>
                </CardTitle>
                <CardDescription>
                    Build your knowledge base by uploading files or pasting text.
                </CardDescription>
                <div className="mt-3 flex flex-wrap gap-2">
                    <InlineHint label="Step 1: Add files or text" detail="PDF, DOCX, Markdown, TXT. Up to 25MB." />
                    <InlineHint label="Step 2: Wait for processing" detail="We parse, OCR if needed, and index automatically." />
                    <InlineHint label="Step 3: Query with coverage" detail="Docs become searchable in the Query Interface." />
                </div>
            </CardHeader>
            <CardContent className="space-y-6 pt-6">
                {/* Premium setup checklist */}


                <div className="flex items-center justify-between rounded-lg border border-border/60 bg-muted/40 px-4 py-3">
                    <div className="space-y-1">
                        <Label htmlFor="sync-ingest" className="text-sm font-medium text-foreground">
                            Index immediately
                        </Label>
                        <p className="text-xs text-muted-foreground">
                            Wait for indexing to finish before returning (recommended for local runs).
                        </p>
                    </div>
                    <input
                        id="sync-ingest"
                        type="checkbox"
                        checked={ingestSync}
                        onChange={event => setIngestSync(event.target.checked)}
                        className="h-4 w-4 accent-primary"
                    />
                </div>

                {/* Upload Zone */}
                <div
                    className={`group flex flex-col items-center justify-center rounded-lg border border-dashed p-8 transition-colors ${isDragOver
                        ? "border-primary/60 bg-primary/10"
                        : "border-border/70 hover:border-border hover:bg-muted/40"
                        }`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                >
                    <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
                        <FolderOpen className="h-5 w-5" />
                    </div>
                    <p className="mt-4 font-semibold text-foreground">
                        {isDragOver ? "Drop files now" : "Click or drag to upload"}
                    </p>
                    <p className="mt-1 text-sm text-muted-foreground">
                        PDF, DOCX, TXT, or Markdown (up to 25MB)
                    </p>
                    <div className="mt-6">
                        <input
                            type="file"
                            accept=".pdf,.doc,.docx,.txt,.md,.markdown"
                            multiple
                            ref={fileInputRef}
                            onChange={handleFileInputChange}
                            className="sr-only"
                            id="document-upload-input"
                        />
                        <Button
                            variant="secondary"
                            onClick={() => fileInputRef.current?.click()}
                            disabled={isUploadingFile || uploadQueue.length > 0}
                            className="px-8"
                        >
                            {isUploadingFile || uploadQueue.length > 0 ? (
                                <>
                                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                                    Processing...
                                </>
                            ) : (
                                "Select Files"
                            )}
                        </Button>
                    </div>
                </div>

                {/* Upload Progress Queue */}
                {uploadQueue.length > 0 && (
                    <div className="rounded-lg border border-border/60 bg-muted/20 p-4">
                        <h4 className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                            Upload Progress
                        </h4>
                        <div className="space-y-3">
                            {uploadQueue.map((item, idx) => (
                                <div key={idx} className="flex items-center gap-3 rounded-md bg-card/60 p-2 text-sm">
                                    {item.status === "done" ? (
                                        <CheckCircle2 className="h-4 w-4 text-primary" />
                                    ) : item.status === "error" ? (
                                        <XCircle className="h-4 w-4 text-destructive" />
                                    ) : item.status === "uploading" || item.status === "processing" ? (
                                        <Loader2 className="h-4 w-4 text-secondary-foreground animate-spin" />
                                    ) : (
                                        <div className="h-4 w-4 rounded-full bg-muted" />
                                    )}
                                    <span className="flex-1 truncate font-medium">{item.filename}</span>
                                    <span className={`text-xs font-semibold uppercase tracking-wide ${item.status === "done" ? "text-primary" :
                                        item.status === "error" ? "text-destructive" :
                                            "text-secondary-foreground"
                                        }`}>
                                        {item.status}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Text Ingest */}
                <div className="space-y-4 rounded-lg border border-border/60 p-4 bg-muted/10">
                    <h4 className="text-sm font-semibold text-foreground">Quick Text Ingest</h4>
                    <div className="grid gap-4 sm:grid-cols-2">
                        <div className="space-y-2">
                            <Label htmlFor="ingestTitle" className="text-xs">Document Title</Label>
                            <Input
                                id="ingestTitle"
                                placeholder="e.g., Company Handbook"
                                value={ingestTitle}
                                onChange={e => setIngestTitle(e.target.value)}
                            />
                        </div>
                        <div className="space-y-2">
                            <Label htmlFor="ingestText" className="text-xs">Content Snippet</Label>
                            <Textarea
                                id="ingestText"
                                rows={3}
                                value={ingestText}
                                onChange={e => setIngestText(e.target.value)}
                                placeholder="Paste content here..."
                                className="resize-none"
                            />
                        </div>
                    </div>
                    <div className="flex justify-end">
                        <Button
                            disabled={isIngesting || !ingestText.trim()}
                            onClick={handleIngest}
                            className="bg-primary text-primary-foreground hover:bg-primary/90"
                            size="sm"
                        >
                            {isIngesting ? "Ingesting..." : "Ingest Text"}
                        </Button>
                    </div>
                </div>

                {/* Premium setup checklist */}
                {documents.length === 0 && (
                    <div className="flex flex-col gap-3 rounded-lg border border-border/70 bg-muted/10 p-4">
                        <div className="flex items-center gap-2">
                            <ShieldCheck className="h-4 w-4 text-primary" />
                            <p className="text-sm font-semibold text-foreground">Getting ready to ingest</p>
                        </div>
                        <div className="space-y-2 text-sm">
                            <div className="flex items-center gap-2">
                                <span className="h-2 w-2 rounded-full bg-amber-500" />
                                <span className="text-foreground">Add at least one document or text snippet</span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="h-2 w-2 rounded-full bg-emerald-500" />
                                <span className="text-foreground">Automatic OCR + indexing will start on upload</span>
                            </div>
                        </div>
                        <div className="text-xs text-muted-foreground">
                            Drag and drop multiple files at once; we’ll queue, parse, and confirm readiness.
                        </div>
                    </div>
                )}
            </CardContent>
        </Card>
    );
}

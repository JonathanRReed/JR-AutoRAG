import { type ChangeEvent, type DragEvent, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { DocumentOut } from "@/types";

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
    documents: DocumentOut[];
    handleDeleteDocument: (id: string, title: string) => void;
    isUploadingFile: boolean;
    uploadFile: (file: File) => Promise<void>;
    fileInputRef: React.RefObject<HTMLInputElement | null>;
    formatDateTime: (value?: string) => string;
}

export function IngestPanel({
    ingestTitle,
    setIngestTitle,
    ingestText,
    setIngestText,
    isIngesting,
    handleIngest,
    documents,
    handleDeleteDocument,
    isUploadingFile,
    uploadFile,
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

        // Initialize queue
        const initialQueue: UploadProgress[] = validFiles.map(f => ({
            filename: f.name,
            status: "pending",
        }));
        setUploadQueue(initialQueue);

        // Process files sequentially
        for (let i = 0; i < validFiles.length; i++) {
            const file = validFiles[i];
            if (!file) continue;
            
            setUploadQueue(prev => prev.map((item, idx) => 
                idx === i ? { ...item, status: "uploading" } : item
            ));

            try {
                await uploadFile(file);
                setUploadQueue(prev => prev.map((item, idx) => 
                    idx === i ? { ...item, status: "done" } : item
                ));
            } catch (err) {
                setUploadQueue(prev => prev.map((item, idx) => 
                    idx === i ? { ...item, status: "error", error: String(err) } : item
                ));
            }
        }

        // Clear queue after a delay
        setTimeout(() => setUploadQueue([]), 3000);
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
        <Card>
            <CardHeader>
                <CardTitle>Ingest Documents</CardTitle>
                <CardDescription>Load PDFs, wikis or notes to make them queryable.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div
                    className={`rounded-lg border-2 border-dashed p-6 text-center transition ${
                        isDragOver 
                            ? "border-blue-500 bg-blue-50" 
                            : "border-border hover:border-foreground"
                    }`}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    onDragLeave={handleDragLeave}
                >
                    <p className="font-medium">
                        {isDragOver ? "Drop files here!" : "Drop PDF, DOCX, TXT, or Markdown files here"}
                    </p>
                    <p className="text-sm text-muted-foreground">
                        Upload multiple files at once. Files are chunked locally before indexing.
                    </p>
                    <div className="mt-3 flex flex-col items-center gap-2">
                        <input
                            type="file"
                            accept=".pdf,.doc,.docx,.txt,.md,.markdown"
                            multiple
                            ref={fileInputRef}
                            onChange={handleFileInputChange}
                            className="sr-only"
                            id="document-upload-input"
                        />
                        <Button variant="outline" onClick={() => fileInputRef.current?.click()} disabled={isUploadingFile || uploadQueue.length > 0}>
                            {isUploadingFile || uploadQueue.length > 0 ? "Processing..." : "Select Files"}
                        </Button>
                        <p className="text-xs text-muted-foreground">Supports PDFs, Word docs, Markdown, and plain text up to 25 MB each.</p>
                    </div>
                </div>

                {/* Upload Progress Queue */}
                {uploadQueue.length > 0 && (
                    <div className="rounded-lg border border-blue-200 bg-blue-50/50 p-3">
                        <p className="mb-2 text-sm font-semibold text-blue-800">Upload Progress</p>
                        <div className="space-y-2">
                            {uploadQueue.map((item, idx) => (
                                <div key={idx} className="flex items-center gap-2 text-sm">
                                    <span className={`h-2 w-2 rounded-full ${
                                        item.status === "done" ? "bg-green-500" :
                                        item.status === "error" ? "bg-red-500" :
                                        item.status === "uploading" ? "bg-blue-500 animate-pulse" :
                                        "bg-gray-300"
                                    }`} />
                                    <span className="flex-1 truncate">{item.filename}</span>
                                    <span className={`text-xs ${
                                        item.status === "done" ? "text-green-600" :
                                        item.status === "error" ? "text-red-600" :
                                        "text-blue-600"
                                    }`}>
                                        {item.status === "done" ? "Done" :
                                         item.status === "error" ? "Failed" :
                                         item.status === "uploading" ? "Uploading..." :
                                         "Pending"}
                                    </span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
                <div className="flex flex-col gap-4 lg:flex-row">
                    <div className="basis-full space-y-2 lg:basis-1/2">
                        <Label htmlFor="ingestTitle">Title</Label>
                        <Input id="ingestTitle" value={ingestTitle} onChange={e => setIngestTitle(e.target.value)} />
                        <Button disabled={isIngesting} onClick={handleIngest}>
                            {isIngesting ? "Ingesting..." : "Ingest Text"}
                        </Button>
                    </div>
                    <div className="basis-full space-y-2 lg:basis-1/2">
                        <Label htmlFor="ingestText">Content</Label>
                        <Textarea
                            id="ingestText"
                            rows={6}
                            value={ingestText}
                            onChange={e => setIngestText(e.target.value)}
                            placeholder="Paste doc text or summary"
                        />
                    </div>
                </div>
                <div>
                    <p className="text-sm font-semibold">Documents ({documents.length})</p>
                    {documents.length === 0 ? (
                        <p className="mt-2 text-sm text-muted-foreground">No documents ingested yet.</p>
                    ) : (
                        <div className="mt-3 grid gap-3 md:grid-cols-2">
                            {documents.map(doc => {
                                const uploadedAt = formatDateTime(doc.metadata?.uploaded_at);
                                const filesize = doc.metadata?.filesize
                                    ? `${(Number(doc.metadata.filesize) / (1024 * 1024)).toFixed(2)} MB`
                                    : undefined;
                                const filename = doc.metadata?.original_filename || doc.metadata?.filename;
                                return (
                                    <div key={doc.id} className="rounded-lg border border-border p-3 text-sm">
                                        <div className="flex items-start justify-between gap-2">
                                            <div>
                                                <p className="font-semibold text-foreground">{doc.title || filename || "Untitled"}</p>
                                                {filename && <p className="text-xs text-muted-foreground">{filename}</p>}
                                            </div>
                                            <Button
                                                variant="ghost"
                                                size="sm"
                                                className="text-destructive hover:bg-destructive/10"
                                                onClick={() => void handleDeleteDocument(doc.id, doc.title || filename || "document")}
                                            >
                                                Remove
                                            </Button>
                                        </div>
                                        <p className="mt-2 text-muted-foreground">{doc.text.slice(0, 140) || "(Empty document)"}…</p>
                                        <div className="mt-3 flex flex-wrap gap-3 text-xs text-muted-foreground">
                                            <span>Uploaded {uploadedAt}</span>
                                            {filesize && <span>Size {filesize}</span>}
                                            {doc.metadata?.content_type && <span>{doc.metadata.content_type}</span>}
                                        </div>
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            </CardContent>
        </Card>
    );
}

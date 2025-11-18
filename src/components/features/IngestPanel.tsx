import { type ChangeEvent, type DragEvent } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { DocumentOut } from "@/types";

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
    uploadFile: (file: File) => void;
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
    const handleFileInputChange = (event: ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            void uploadFile(file);
        }
    };

    const handleDrop = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
        const file = event.dataTransfer.files?.[0];
        if (file) {
            void uploadFile(file);
        }
    };

    const handleDragOver = (event: DragEvent<HTMLDivElement>) => {
        event.preventDefault();
    };

    return (
        <Card>
            <CardHeader>
                <CardTitle>Ingest Documents</CardTitle>
                <CardDescription>Load PDFs, wikis or notes to make them queryable.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                <div
                    className="rounded-lg border border-dashed border-border p-6 text-center transition hover:border-foreground"
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                >
                    <p className="font-medium">Drop a PDF, DOCX, or TXT here</p>
                    <p className="text-sm text-muted-foreground">Files are chunked locally before indexing.</p>
                    <div className="mt-3 flex flex-col items-center gap-2">
                        <input
                            type="file"
                            accept=".pdf,.doc,.docx,.txt"
                            ref={fileInputRef}
                            onChange={handleFileInputChange}
                            className="sr-only"
                            id="document-upload-input"
                        />
                        <Button variant="outline" onClick={() => fileInputRef.current?.click()} disabled={isUploadingFile}>
                            {isUploadingFile ? "Uploading..." : "Select File"}
                        </Button>
                        <p className="text-xs text-muted-foreground">Supports PDFs, Word docs, and plain text up to 25 MB.</p>
                    </div>
                </div>
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

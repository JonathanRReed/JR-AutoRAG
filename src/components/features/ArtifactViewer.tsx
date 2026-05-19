import React, { useState, useEffect } from 'react';
import { X, Network, Layers, Database, ChevronRight, ChevronDown, FileText, Info } from 'lucide-react';
import { Button } from '../ui/button';
import { buildApiUrl } from '@/lib/api-url';

interface ArtifactViewerProps {
    type: 'graph_rag' | 'raptor';
    onClose: () => void;
    baseUrl?: string;
    apiKey?: string;
}

export function ArtifactViewer({ type, onClose, baseUrl, apiKey = "" }: ArtifactViewerProps) {
    const [data, setData] = useState<any>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                const endpoint = type === 'graph_rag' ? '/api/artifacts/graph' : '/api/artifacts/raptor';
                const headers: Record<string, string> = {};
                const trimmedApiKey = apiKey.trim();
                if (trimmedApiKey) {
                    headers["X-API-Key"] = trimmedApiKey;
                }
                const res = await fetch(buildApiUrl(baseUrl, endpoint), {
                    headers: Object.keys(headers).length ? headers : undefined,
                });
                if (!res.ok) throw new Error("Failed to fetch artifact data");
                const json = await res.json();
                setData(json);
            } catch (e: any) {
                setError(e.message);
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [type, baseUrl, apiKey]);

    return (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-background/80 backdrop-blur-sm animate-in fade-in duration-200">
            <div className="relative flex h-[85vh] w-[90vw] max-w-6xl flex-col overflow-hidden rounded-2xl border border-border bg-card shadow-2xl animate-in zoom-in-95 duration-200">
                {/* Header */}
                <div className="flex items-center justify-between border-b border-border/40 bg-muted/20 px-6 py-4">
                    <div className="flex items-center gap-3">
                        <div className={`flex h-10 w-10 items-center justify-center rounded-xl ${type === 'graph_rag' ? 'bg-primary/10 text-primary' : 'bg-secondary/10 text-foreground'}`}>
                            {type === 'graph_rag' ? <Network size={24} /> : <Layers size={24} />}
                        </div>
                        <div>
                            <h2 className="text-xl font-bold tracking-tight">{type === 'graph_rag' ? 'Knowledge Graph Explorer' : 'Hierarchical Document Tree'}</h2>
                            <p className="text-sm text-muted-foreground">{type === 'graph_rag' ? 'Entities, relationships, and community clusters' : 'Navigable RAPTOR summary levels'}</p>
                        </div>
                    </div>
                    <Button variant="ghost" size="icon" onClick={onClose} className="rounded-full hover:bg-muted">
                        <X size={20} />
                    </Button>
                </div>

                {/* Content Area */}
                <div className="flex-1 overflow-hidden">
                    {loading ? (
                        <div className="flex h-full items-center justify-center">
                            <div className="flex flex-col items-center gap-4">
                                <div className="h-12 w-12 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
                                <p className="text-sm font-medium animate-pulse">Building Visualization...</p>
                            </div>
                        </div>
                    ) : error ? (
                        <div className="flex h-full items-center justify-center p-12 text-center">
                            <div className="max-w-md space-y-4">
                                <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-full bg-destructive/10 text-destructive">
                                    <Info size={32} />
                                </div>
                                <h3 className="text-lg font-semibold">Failed to load artifact</h3>
                                <p className="text-muted-foreground">{error}</p>
                                <Button onClick={onClose}>Close Viewer</Button>
                            </div>
                        </div>
                    ) : (
                        <div className="h-full overflow-y-auto p-6">
                            {type === 'graph_rag' ? (
                                <GraphVisualization data={data} />
                            ) : (
                                <RaptorVisualization data={data} />
                            )}
                        </div>
                    )}
                </div>

                {/* Footer info */}
                <div className="border-t border-border/40 bg-muted/5 px-6 py-3 text-[10px] text-muted-foreground flex justify-between items-center">
                    <span>G4 Engine - Deep Artifact Visibility</span>
                    {data && (
                        <div className="flex gap-4">
                            {type === 'graph_rag' ? (
                                <>
                                    <span>Entities: {data.entities?.length || 0}</span>
                                    <span>Relationships: {data.relationships?.length || 0}</span>
                                    <span>Communities: {data.communities?.length || 0}</span>
                                </>
                            ) : (
                                <span>Documents: {Object.keys(data.trees || {}).length}</span>
                            )}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

function GraphVisualization({ data }: { data: any }) {
    const [activeTab, setActiveTab] = useState<'entities' | 'communities' | 'relationships'>('entities');

    return (
        <div className="space-y-6">
            <div className="flex gap-1 rounded-lg bg-muted/30 p-1 w-fit">
                {(['entities', 'relationships', 'communities'] as const).map(tab => (
                    <button
                        key={tab}
                        onClick={() => setActiveTab(tab)}
                        className={`px-4 py-1.5 text-xs font-medium rounded-md capitalize transition-all ${activeTab === tab ? 'bg-background text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}
                    >
                        {tab}
                    </button>
                ))}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {activeTab === 'entities' && data.entities?.map((e: any, i: number) => (
                    <div key={i} className="group rounded-xl border border-border/50 bg-muted/10 p-4 hover:border-primary/30 transition-all">
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-primary px-2 py-0.5 rounded bg-primary/5">{e.type || 'Entity'}</span>
                        </div>
                        <h4 className="font-bold text-foreground group-hover:text-primary transition-colors">{e.name}</h4>
                        <p className="mt-2 text-xs text-muted-foreground leading-relaxed line-clamp-3">{e.description}</p>
                    </div>
                ))}

                {activeTab === 'communities' && data.communities?.map((c: any, i: number) => (
                    <div key={i} className="rounded-xl border border-border/50 bg-muted/10 p-4 overflow-hidden">
                        <div className="flex items-center justify-between mb-3">
                            <span className="text-xs font-bold text-primary">Cluster #{c.id}</span>
                            <span className="text-[10px] text-muted-foreground">{c.entities?.length || 0} entities</span>
                        </div>
                        <p className="text-xs text-foreground line-clamp-3 mb-4 italic">"{c.summary}"</p>
                        <div className="flex flex-wrap gap-1.5">
                            {c.entities?.slice(0, 5).map((ent: string, j: number) => (
                                <span key={j} className="text-[9px] bg-muted text-foreground px-1.5 py-0.5 rounded border border-border">{ent}</span>
                            ))}
                            {c.entities?.length > 5 && <span className="text-[9px] text-muted-foreground px-1.5 py-0.5">+{c.entities.length - 5} more</span>}
                        </div>
                    </div>
                ))}

                {activeTab === 'relationships' && (
                    <div className="col-span-full overflow-hidden rounded-xl border border-border/50 bg-muted/10">
                        <table className="w-full text-left text-xs">
                            <thead className="bg-muted/30 text-muted-foreground">
                                <tr>
                                    <th className="px-4 py-3 font-semibold">Source</th>
                                    <th className="px-4 py-3 font-semibold">Relationship</th>
                                    <th className="px-4 py-3 font-semibold">Target</th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-border/30">
                                {data.relationships?.map((r: any, i: number) => (
                                    <tr key={i} className="hover:bg-muted/20 transition-colors">
                                        <td className="px-4 py-3 font-medium text-primary">{r.source}</td>
                                        <td className="px-4 py-3 text-muted-foreground max-w-xs truncate" title={r.description}>{r.description}</td>
                                        <td className="px-4 py-3 font-medium text-foreground">{r.target}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
}

function RaptorVisualization({ data }: { data: any }) {
    const trees = data.trees || {};
    const [selectedDoc, setSelectedDoc] = useState<string | null>(Object.keys(trees)[0] || null);

    if (!selectedDoc) return <div className="text-center py-20 text-muted-foreground">No document hierarchies built yet.</div>;

    const currentTree = trees[selectedDoc];
    const rootNode = currentTree.nodes.find((n: any) => n.id === currentTree.root_id);

    return (
        <div className="flex flex-col h-full gap-6">
            <div className="flex flex-wrap gap-2">
                {Object.keys(trees).map(docId => (
                    <button
                        key={docId}
                        onClick={() => setSelectedDoc(docId)}
                        className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs transition-all ${selectedDoc === docId ? 'bg-primary/10 border-primary/50 text-foreground ring-1 ring-primary/20' : 'bg-muted/20 border-border/40 text-muted-foreground hover:bg-muted/40'}`}
                    >
                        <FileText size={14} />
                        <span className="truncate max-w-[150px]">{docId}</span>
                    </button>
                ))}
            </div>

            <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 min-h-0">
                <div className="lg:col-span-1 border-r border-border/40 pr-6 overflow-y-auto space-y-4">
                    <h3 className="text-xs font-semibold uppercase tracking-widest text-muted-foreground flex items-center gap-2">
                        <Layers size={14} /> Tree Structure
                    </h3>
                    <div className="space-y-1">
                        {currentTree.nodes.sort((a: any, b: any) => a.level - b.level).map((node: any) => (
                            <TreeNode key={node.id} node={node} />
                        ))}
                    </div>
                </div>

                <div className="lg:col-span-3 space-y-6 overflow-y-auto pb-8">
                    <div className="rounded-2xl border border-primary/20 bg-primary/5 p-6 backdrop-blur-sm">
                        <h3 className="text-lg font-bold text-primary flex items-center gap-2 mb-4">
                            <Database size={20} /> Corporate Intelligence Summary
                        </h3>
                        <p className="text-sm leading-relaxed text-foreground whitespace-pre-wrap">{rootNode?.summary || "Summary processing..."}</p>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        {currentTree.nodes.filter((n: any) => n.level > 0).map((node: any) => (
                            <div key={node.id} className="rounded-xl border border-border/40 bg-muted/10 p-4 hover:border-primary/30 transition-all">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="text-[10px] font-mono text-primary">LEVEL {node.level} - {node.id}</span>
                                </div>
                                <p className="text-xs text-muted-foreground line-clamp-4">{node.summary}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    );
}

function TreeNode({ node }: { node: any }) {
    return (
        <div className="flex items-center gap-2 py-1.5 px-2 hover:bg-muted/40 rounded transition-colors cursor-pointer group" style={{ paddingLeft: `${node.level * 12 + 8}px` }}>
            <div className={`h-1.5 w-1.5 rounded-full ${node.level === 0 ? 'bg-primary shadow-[0_0_8px_var(--primary)]' : 'bg-muted-foreground/30'}`} />
            <span className={`text-[11px] truncate ${node.level === 0 ? 'font-bold text-foreground' : 'text-muted-foreground group-hover:text-foreground'}`}>
                {node.title || `Level ${node.level} Summary`}
            </span>
            {node.children?.length > 0 && <span className="text-[9px] text-muted-foreground/50 ml-auto">{node.children.length}</span>}
        </div>
    );
}

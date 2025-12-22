import React from "react";
import { Shield, Globe } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip";
import type { QueryMode } from "@/types";

interface QueryModeToggleProps {
    mode: QueryMode;
    onChange: (mode: QueryMode) => void;
    disabled?: boolean;
}

/**
 * Toggle between Grounded and Open Domain query modes.
 * 
 * Grounded: Only answer from corpus documents
 * Open Domain: LLM can use general knowledge
 */
export function QueryModeToggle({ mode, onChange, disabled }: QueryModeToggleProps) {
    const isGrounded = mode === "grounded";

    return (
        <TooltipProvider>
            <div className="flex items-center gap-1 p-0.5 rounded-lg bg-muted/50 border border-border/50">
                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button
                            variant={isGrounded ? "default" : "ghost"}
                            size="sm"
                            className={`h-7 px-2.5 gap-1.5 text-xs font-medium transition-all ${isGrounded
                                    ? "bg-primary text-primary-foreground shadow-sm"
                                    : "text-muted-foreground hover:text-foreground"
                                }`}
                            onClick={() => onChange("grounded")}
                            disabled={disabled}
                        >
                            <Shield className="h-3.5 w-3.5" />
                            <span>Grounded</span>
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="max-w-[200px]">
                        <p className="text-xs">
                            <strong>Grounded Mode:</strong> Answers only from your documents.
                            If no evidence found, returns helpful suggestions instead of guessing.
                        </p>
                    </TooltipContent>
                </Tooltip>

                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button
                            variant={!isGrounded ? "default" : "ghost"}
                            size="sm"
                            className={`h-7 px-2.5 gap-1.5 text-xs font-medium transition-all ${!isGrounded
                                    ? "bg-primary text-primary-foreground shadow-sm"
                                    : "text-muted-foreground hover:text-foreground"
                                }`}
                            onClick={() => onChange("open_domain")}
                            disabled={disabled}
                        >
                            <Globe className="h-3.5 w-3.5" />
                            <span>Open</span>
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="max-w-[200px]">
                        <p className="text-xs">
                            <strong>Open Domain:</strong> LLM can use general knowledge
                            when corpus evidence is insufficient.
                        </p>
                    </TooltipContent>
                </Tooltip>
            </div>
        </TooltipProvider>
    );
}

import React from "react";
import { Shield, Globe } from "lucide-react";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
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

export function QueryModeToggle({ mode, onChange, disabled }: QueryModeToggleProps) {
    const isGrounded = mode === "grounded";

    return (
        <TooltipProvider>
            <ToggleGroup aria-label="Query mode">
                <Tooltip>
                    <TooltipTrigger asChild>
                        <ToggleGroupItem
                            pressed={isGrounded}
                            onClick={() => onChange("grounded")}
                            disabled={disabled}
                        >
                            <Shield data-icon="inline-start" />
                            <span>Grounded</span>
                        </ToggleGroupItem>
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
                        <ToggleGroupItem
                            pressed={!isGrounded}
                            onClick={() => onChange("open_domain")}
                            disabled={disabled}
                        >
                            <Globe data-icon="inline-start" />
                            <span>Open</span>
                        </ToggleGroupItem>
                    </TooltipTrigger>
                    <TooltipContent side="bottom" className="max-w-[200px]">
                        <p className="text-xs">
                            <strong>Open Domain:</strong> LLM can use general knowledge
                            when corpus evidence is insufficient.
                        </p>
                    </TooltipContent>
                </Tooltip>
            </ToggleGroup>
        </TooltipProvider>
    );
}

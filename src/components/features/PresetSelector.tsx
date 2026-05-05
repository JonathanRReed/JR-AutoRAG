import { useMemo } from "react";
import { Zap, Rocket, Scale, Search, Target, Info } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { PresetLevel } from "@/types";
import { PRESET_DEFINITIONS } from "@/types";

interface PresetSelectorProps {
    value: PresetLevel;
    onChange: (preset: PresetLevel) => void;
    disabled?: boolean;
    compact?: boolean;
}

const PRESET_ICONS: Record<PresetLevel, typeof Zap> = {
    turbo: Zap,
    fast: Rocket,
    balanced: Scale,
    thorough: Search,
    ultra_accurate: Target,
};

const PRESET_COLORS: Record<PresetLevel, string> = {
    turbo: "text-muted-foreground",
    fast: "text-muted-foreground",
    balanced: "text-primary",
    thorough: "text-foreground",
    ultra_accurate: "text-primary",
};

export function PresetSelector({ value, onChange, disabled, compact }: PresetSelectorProps) {
    const presets = useMemo(() => PRESET_DEFINITIONS, []);

    if (compact) {
        return (
            <div className="flex flex-col gap-2">
                <div className="flex items-center justify-between">
                    <div>
                        <h4 className="text-xs font-semibold text-muted-foreground uppercase tracking-wider">
                            Advanced Engineering Presets
                        </h4>
                        <p className="text-[10px] text-muted-foreground/70">
                            Apply tuned defaults for common workflows
                        </p>
                    </div>
                </div>
                <div className="flex items-center gap-1 p-1 bg-muted/50 rounded-lg">
                    {presets.map((preset) => {
                        const Icon = PRESET_ICONS[preset.level];
                        const isActive = value === preset.level;
                        return (
                            <Button
                                key={preset.level}
                                variant={isActive ? "default" : "ghost"}
                                size="sm"
                                onClick={() => onChange(preset.level)}
                                disabled={disabled}
                                className={`h-8 px-2 ${isActive ? "" : "hover:bg-muted"}`}
                                title={`${preset.name}: ${preset.description}`}
                            >
                                <Icon className={`h-4 w-4 ${isActive ? "" : PRESET_COLORS[preset.level]}`} />
                                {isActive && <span className="ml-1 text-xs">{preset.name}</span>}
                            </Button>
                        );
                    })}
                </div>
            </div>
        );
    }

    return (
        <div className="flex flex-col gap-2">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Info className="h-4 w-4" />
                <span>Speed to accuracy</span>
            </div>
            <div className="grid grid-cols-5 gap-2">
                {presets.map((preset) => {
                    const Icon = PRESET_ICONS[preset.level];
                    const isActive = value === preset.level;
                    return (
                        <button
                            key={preset.level}
                            onClick={() => onChange(preset.level)}
                            disabled={disabled}
                            className={`
                                relative flex flex-col items-center gap-1 p-3 rounded-lg border transition-all
                                ${isActive
                                    ? "border-primary bg-primary/10 shadow-md"
                                    : "border-border hover:border-primary/50 hover:bg-muted/50"
                                }
                                ${disabled ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}
                            `}
                        >
                            <Icon className={`h-5 w-5 ${isActive ? "text-primary" : PRESET_COLORS[preset.level]}`} />
                            <span className={`text-xs font-medium ${isActive ? "text-primary" : "text-foreground"}`}>
                                {preset.name}
                            </span>
                            <span className="text-[10px] text-muted-foreground text-center leading-tight">
                                {preset.description}
                            </span>
                            {isActive && (
                                <div className="absolute -top-1 -right-1 h-3 w-3 bg-primary rounded-full border-2 border-background" />
                            )}
                        </button>
                    );
                })}
            </div>
            {/* Feature badges for selected preset */}
            <div className="flex flex-wrap gap-1 mt-2">
                {presets.find(p => p.level === value)?.features.map((feature) => (
                    <span
                        key={feature}
                        className="text-[10px] px-2 py-0.5 bg-muted rounded-full text-muted-foreground"
                    >
                        {feature}
                    </span>
                ))}
            </div>
        </div>
    );
}

export default PresetSelector;

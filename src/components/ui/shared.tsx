import { type ReactNode } from "react";
import { Loader2 } from "lucide-react";

interface PageSectionProps {
    title: string;
    description?: string;
    icon?: ReactNode;
    actions?: ReactNode;
    children: ReactNode;
}

export function PageSection({ title, description, icon, actions, children }: PageSectionProps) {
    return (
        <section className="space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-3">
                    {icon && (
                        <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10 text-primary">
                            {icon}
                        </div>
                    )}
                    <div>
                        <h2 className="text-lg font-semibold text-foreground">{title}</h2>
                        {description && (
                            <p className="text-sm text-muted-foreground">{description}</p>
                        )}
                    </div>
                </div>
                {actions && <div className="flex items-center gap-2">{actions}</div>}
            </div>
            {children}
        </section>
    );
}

interface StatCardProps {
    title: string;
    value: string | number;
    subtitle?: string;
    icon?: ReactNode;
    trend?: "up" | "down" | "neutral";
    className?: string;
}

export function StatCard({ title, value, subtitle, icon, trend, className = "" }: StatCardProps) {
    const trendStyles = {
        up: "text-primary",
        down: "text-destructive",
        neutral: "text-muted-foreground",
    };

    return (
        <div className={`rounded-lg border border-border/60 bg-card p-4 transition-colors ${className}`}>
            <div className="flex items-start justify-between">
                <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                    {title}
                </p>
                {icon && <span className="text-muted-foreground">{icon}</span>}
            </div>
            <p className="mt-2 text-2xl font-bold text-foreground">{value}</p>
            {subtitle && (
                <p className={`mt-1 text-xs ${trend ? trendStyles[trend] : "text-muted-foreground"}`}>
                    {subtitle}
                </p>
            )}
        </div>
    );
}

interface EmptyStateProps {
    icon: ReactNode;
    title: string;
    description?: string;
    action?: ReactNode;
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
    return (
        <div className="flex flex-col items-center justify-center rounded-lg border border-dashed border-border/70 bg-muted/10 p-12 text-center">
            <div className="flex h-12 w-12 items-center justify-center rounded-full bg-muted text-muted-foreground">
                {icon}
            </div>
            <h3 className="mt-4 font-medium text-foreground">{title}</h3>
            {description && (
                <p className="mt-1 max-w-sm text-sm text-muted-foreground">{description}</p>
            )}
            {action && <div className="mt-4">{action}</div>}
        </div>
    );
}

interface LoadingStateProps {
    message?: string;
}

export function LoadingState({ message = "Loading..." }: LoadingStateProps) {
    return (
        <div className="flex items-center justify-center gap-3 rounded-lg border border-border/60 bg-card p-8">
            <Loader2 className="h-5 w-5 animate-spin text-primary" />
            <span className="text-sm text-muted-foreground">{message}</span>
        </div>
    );
}

interface StatusBadgeProps {
    status: "success" | "warning" | "error" | "info" | "neutral";
    children: ReactNode;
    dot?: boolean;
}

export function StatusBadge({ status, children, dot = false }: StatusBadgeProps) {
    const styles = {
        success: "bg-primary/10 text-primary",
        warning: "bg-secondary/30 text-secondary-foreground",
        error: "bg-destructive/20 text-destructive",
        info: "bg-secondary/20 text-secondary-foreground",
        neutral: "bg-muted text-muted-foreground",
    };

    const dotStyles = {
        success: "bg-primary",
        warning: "bg-secondary",
        error: "bg-destructive",
        info: "bg-secondary",
        neutral: "bg-muted-foreground",
    };

    return (
        <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${styles[status]}`}>
            {dot && <span className={`h-1.5 w-1.5 rounded-full ${dotStyles[status]}`} />}
            {children}
        </span>
    );
}

interface InfoRowProps {
    label: string;
    value: ReactNode;
    className?: string;
}

export function InfoRow({ label, value, className = "" }: InfoRowProps) {
    return (
        <div className={`flex items-center justify-between gap-4 ${className}`}>
            <span className="text-sm text-muted-foreground">{label}</span>
            <span className="text-sm font-medium text-foreground">{value}</span>
        </div>
    );
}

interface SectionDividerProps {
    label?: string;
}

export function SectionDivider({ label }: SectionDividerProps) {
    if (!label) {
        return <hr className="border-border" />;
    }
    return (
        <div className="flex items-center gap-4">
            <hr className="flex-1 border-border" />
            <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                {label}
            </span>
            <hr className="flex-1 border-border" />
        </div>
    );
}

interface ProgressBarProps {
    value: number;
    max?: number;
    label?: string;
    showValue?: boolean;
    size?: "sm" | "md" | "lg";
}

export function ProgressBar({ value, max = 100, label, showValue = true, size = "md" }: ProgressBarProps) {
    const percentage = Math.min((value / max) * 100, 100);
    const barColor = percentage > 80 ? "bg-primary" : percentage > 50 ? "bg-secondary" : percentage > 20 ? "bg-secondary/70" : "bg-destructive";
    const heights = { sm: "h-1", md: "h-2", lg: "h-3" };

    return (
        <div className="space-y-1">
            {(label || showValue) && (
                <div className="flex justify-between text-xs">
                    {label && <span className="text-muted-foreground">{label}</span>}
                    {showValue && <span className="font-medium text-foreground">{percentage.toFixed(0)}%</span>}
                </div>
            )}
            <div className={`overflow-hidden rounded-full bg-muted ${heights[size]}`}>
                <div
                    className={`${heights[size]} rounded-full transition-all duration-500 ${barColor}`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    );
}

interface FeatureToggleProps {
    label: string;
    description?: string;
    icon?: ReactNode;
    enabled: boolean;
    onChange: (enabled: boolean) => void;
}

export function FeatureToggle({ label, description, icon, enabled, onChange }: FeatureToggleProps) {
    return (
        <button
            type="button"
            onClick={() => onChange(!enabled)}
            className={`flex w-full items-start gap-3 rounded-lg border p-4 text-left transition-colors ${
                enabled
                    ? "border-primary/50 bg-primary/5"
                    : "border-border/60 bg-card hover:border-border"
            }`}
        >
            {icon && (
                <div className={`flex h-8 w-8 items-center justify-center rounded-md ${
                    enabled ? "bg-primary/20 text-primary" : "bg-muted text-muted-foreground"
                }`}>
                    {icon}
                </div>
            )}
            <div className="flex-1">
                <p className="font-medium text-foreground">{label}</p>
                {description && (
                    <p className="mt-0.5 text-xs text-muted-foreground">{description}</p>
                )}
            </div>
            <div className={`relative h-6 w-11 rounded-full border border-border/50 transition-colors ${
                enabled ? "bg-primary" : "bg-muted"
            }`}>
                <div className={`absolute top-1 h-4 w-4 rounded-full bg-white shadow-sm transition-transform ${
                    enabled ? "translate-x-6" : "translate-x-1"
                }`} />
            </div>
        </button>
    );
}

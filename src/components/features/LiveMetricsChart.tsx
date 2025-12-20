import { useMemo } from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface SparklineProps {
    data: number[];
    height?: number;
    className?: string;
}

export function Sparkline({ data, height = 24, className = "" }: SparklineProps) {
    const normalizedData = useMemo(() => {
        if (data.length === 0) return [];
        const max = Math.max(...data, 1);
        return data.map(v => (v / max) * 100);
    }, [data]);

    if (normalizedData.length === 0) {
        return (
            <div className={`sparkline ${className}`} style={{ height }}>
                <div className="sparkline-bar" style={{ height: "20%" }} />
                <div className="sparkline-bar" style={{ height: "20%" }} />
                <div className="sparkline-bar" style={{ height: "20%" }} />
            </div>
        );
    }

    return (
        <div className={`sparkline ${className}`} style={{ height }}>
            {normalizedData.map((value, idx) => (
                <div
                    key={idx}
                    className="sparkline-bar"
                    style={{ height: `${Math.max(value, 5)}%` }}
                />
            ))}
        </div>
    );
}

interface GaugeRingProps {
    value: number; // 0-100
    size?: number;
    strokeWidth?: number;
    className?: string;
    showValue?: boolean;
}

export function GaugeRing({
    value,
    size = 48,
    strokeWidth = 4,
    className = "",
    showValue = true,
}: GaugeRingProps) {
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (value / 100) * circumference;

    return (
        <div className={`gauge-ring ${className}`} style={{ width: size, height: size }}>
            <svg width={size} height={size}>
                <circle
                    className="gauge-ring-bg"
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    strokeWidth={strokeWidth}
                />
                <circle
                    className="gauge-ring-fill"
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    strokeWidth={strokeWidth}
                    strokeDasharray={circumference}
                    strokeDashoffset={offset}
                />
            </svg>
            {showValue && (
                <span className="gauge-ring-value">
                    {value.toFixed(0)}%
                </span>
            )}
        </div>
    );
}

interface TrendIndicatorProps {
    current: number;
    previous: number;
    format?: (v: number) => string;
    inverted?: boolean; // If true, lower is better (e.g., latency)
}

export function TrendIndicator({
    current,
    previous,
    format = v => v.toFixed(1),
    inverted = false,
}: TrendIndicatorProps) {
    const diff = current - previous;
    const percentChange = previous !== 0 ? (diff / previous) * 100 : 0;

    const isPositive = inverted ? diff < 0 : diff > 0;
    const isNeutral = Math.abs(percentChange) < 1;

    if (isNeutral) {
        return (
            <span className="trend-indicator trend-neutral">
                <Minus className="h-3 w-3" />
                <span>No change</span>
            </span>
        );
    }

    return (
        <span className={`trend-indicator ${isPositive ? "trend-up" : "trend-down"}`}>
            {isPositive ? (
                <TrendingUp className="h-3 w-3" />
            ) : (
                <TrendingDown className="h-3 w-3" />
            )}
            <span>{Math.abs(percentChange).toFixed(1)}%</span>
        </span>
    );
}

interface AnimatedStatProps {
    value: number;
    format?: (v: number) => string;
    className?: string;
}

export function AnimatedStat({ value, format = v => v.toFixed(0), className = "" }: AnimatedStatProps) {
    return (
        <span className={`stat-value animate-count-up ${className}`}>
            {format(value)}
        </span>
    );
}

interface StatCardProps {
    title: string;
    value: string | number;
    subtitle?: string;
    trend?: { current: number; previous: number; inverted?: boolean };
    icon?: React.ReactNode;
    sparklineData?: number[];
    gauge?: number;
    className?: string;
}

export function StatCard({
    title,
    value,
    subtitle,
    trend,
    icon,
    sparklineData,
    gauge,
    className = "",
}: StatCardProps) {
    return (
        <div className={`stat-card glass-card rounded-lg p-5 ${className}`}>
            <div className="flex items-start justify-between gap-3">
                <div className="flex-1 min-w-0">
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                        {title}
                    </p>
                    <p className="mt-2 text-3xl font-bold text-foreground stat-value">
                        {value}
                    </p>
                    {subtitle && (
                        <p className="mt-1.5 text-xs text-muted-foreground">
                            {subtitle}
                        </p>
                    )}
                    {trend && (
                        <div className="mt-2">
                            <TrendIndicator {...trend} />
                        </div>
                    )}
                </div>
                <div className="flex flex-col items-end gap-2">
                    {icon && (
                        <span className="text-muted-foreground/60">
                            {icon}
                        </span>
                    )}
                    {gauge !== undefined && (
                        <GaugeRing value={gauge} size={40} strokeWidth={3} />
                    )}
                    {sparklineData && sparklineData.length > 0 && (
                        <Sparkline data={sparklineData} height={20} />
                    )}
                </div>
            </div>
        </div>
    );
}

interface DurationBarProps {
    label: string;
    value: number;
    max: number;
    className?: string;
}

export function DurationBar({ label, value, max, className = "" }: DurationBarProps) {
    const percentage = max > 0 ? Math.min((value / max) * 100, 100) : 0;
    const durationClass = percentage > 80 ? "slow" : percentage > 50 ? "medium" : "fast";

    return (
        <div className={`space-y-1 ${className}`}>
            <div className="flex justify-between text-xs">
                <span className="text-muted-foreground truncate">{label}</span>
                <span className="font-medium text-foreground shrink-0">{value.toFixed(0)}ms</span>
            </div>
            <div className="duration-bar">
                <div
                    className={`duration-bar-fill ${durationClass}`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
        </div>
    );
}

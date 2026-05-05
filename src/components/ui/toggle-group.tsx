import * as React from "react";

import { cn } from "@/lib/utils";

function ToggleGroup({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="toggle-group"
      role="group"
      className={cn("inline-flex items-center gap-1 rounded-lg bg-muted/40 p-1", className)}
      {...props}
    />
  );
}

function ToggleGroupItem({
  className,
  pressed = false,
  ...props
}: React.ComponentProps<"button"> & {
  pressed?: boolean;
}) {
  return (
    <button
      type="button"
      data-slot="toggle-group-item"
      aria-pressed={pressed}
      className={cn(
        "inline-flex h-8 items-center justify-center gap-2 rounded-md px-3 text-sm font-medium text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50 disabled:pointer-events-none disabled:opacity-50",
        pressed && "bg-card text-foreground shadow-sm",
        className,
      )}
      {...props}
    />
  );
}

export { ToggleGroup, ToggleGroupItem };

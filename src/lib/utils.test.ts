import { describe, expect, test } from "bun:test";

import { cn } from "./utils";

describe("cn", () => {
  test("merges only truthy class names", () => {
    expect(cn("p-2", undefined, "text-lg", "", false && "hidden")).toBe("p-2 text-lg");
  });

  test("deduplicates classes with tailwind merge semantics", () => {
    expect(cn("p-2", "p-4", "p-2")).toBe("p-2");
  });
});

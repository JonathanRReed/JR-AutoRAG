import { readFileSync } from "node:fs";
import { describe, expect, test } from "bun:test";

const css = readFileSync(new URL("../../styles/globals.css", import.meta.url), "utf8");
const luminance = (hex: string) => {
  const rgb = hex.match(/[a-f\d]{2}/gi)!.map(value => parseInt(value, 16) / 255)
    .map(value => value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4);
  return rgb[0]! * 0.2126 + rgb[1]! * 0.7152 + rgb[2]! * 0.0722;
};

describe("theme text contrast", () => {
  for (const theme of [":root", ".dark"]) {
    const textPairs = [
      ["primary", "primary-foreground"],
      ["background", "foreground"],
      ["card", "card-foreground"],
      ["background", "muted-foreground"],
    ] as const;
    for (const [background, foreground] of textPairs) {
      test(`${theme} ${foreground} text meets WCAG AA`, () => {
        const block = css.slice(css.indexOf(`${theme} {`)).split("}")[0]!;
        const token = (name: string) => block.match(new RegExp(`--${name}:\\s*(#[a-f\\d]{6})`, "i"))![1]!;
        const values = [luminance(token(background)), luminance(token(foreground))].sort((a, b) => a - b);
        expect((values[1]! + 0.05) / (values[0]! + 0.05)).toBeGreaterThanOrEqual(4.5);
      });
    }
  }
});

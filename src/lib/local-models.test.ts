import { expect, test } from "bun:test";
import { selectLocalChatModel } from "./local-models";

test("does not select known embedding or reranking families for chat", () => {
  expect(selectLocalChatModel(["bge-m3:latest", "all-minilm:latest", "qwen3-embedding:4b", "bge-reranker-v2-m3", "qwen3.5:4b"], [])).toBe("qwen3.5:4b");
  expect(selectLocalChatModel(["embeddinggemma", "qwen3-embedding:4b"], [])).toBe("");
});

test("prefers an already running valid model without inventing a download", () => {
  expect(selectLocalChatModel(["qwen3.5:9b", "my-custom-chat"], ["my-custom-chat"])).toBe("my-custom-chat");
  expect(selectLocalChatModel(["legacy-chat"], ["deleted-model"])).toBe("legacy-chat");
});

test("uses a modern installed fallback, leaving custom names intact", () => {
  expect(selectLocalChatModel(["llama3:latest", "qwen3.5:4b"], [])).toBe("qwen3.5:4b");
  expect(selectLocalChatModel(["My Custom Chat:Q4"], [])).toBe("My Custom Chat:Q4");
});

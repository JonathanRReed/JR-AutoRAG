// Discovery order is not a quality ranking and may list embedders first.
export const isLikelyChatModel = (model: string): boolean => !/embed|rerank|(?:^|\/)(?:bge-|all-minilm|gte-|e5-)/i.test(model);

export function selectLocalChatModel(models: string[], running: string[]): string {
  const candidates = models.filter(isLikelyChatModel);
  const active = running.find(model => candidates.includes(model));
  if (active) return active;
  for (const preferred of ["qwen3.5:4b", "qwen3.5:9b", "gemma4:e4b", "gemma4:e2b"]) {
    if (candidates.includes(preferred)) return preferred;
  }
  return candidates[0] ?? "";
}

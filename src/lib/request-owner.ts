export class RequestOwner {
  private active = true;
  private readonly controllers = new Set<AbortController>();
  private readonly named = new Map<string, AbortController>();

  activate(): void { this.active = true; }
  close(): void {
    this.active = false;
    for (const controller of this.controllers) controller.abort();
    this.controllers.clear();
    this.named.clear();
  }

  start(key?: string) {
    if (key) {
      const previous = this.named.get(key);
      previous?.abort();
      if (previous) this.controllers.delete(previous);
    }
    const controller = new AbortController();
    if (key) this.named.set(key, controller);
    this.controllers.add(controller);
    const current = () => this.active && !controller.signal.aborted;
    const check = () => { if (!current()) throw new DOMException("Connection changed", "AbortError"); };
    return {
      current,
      abort: () => controller.abort(),
      finish: () => {
        controller.abort();
        this.controllers.delete(controller);
        if (key && this.named.get(key) === controller) this.named.delete(key);
      },
      fetch: async (input: RequestInfo | URL, init: Omit<RequestInit, "signal"> = {}) => {
        check();
        const response = await fetch(input, { ...init, signal: controller.signal });
        check();
        return response;
      },
      json: async <T = unknown>(response: Response): Promise<T> => {
        const data = await response.json() as T;
        check();
        return data;
      },
      blob: async (response: Response): Promise<Blob> => {
        const data = await response.blob();
        check();
        return data;
      },
    };
  }
}

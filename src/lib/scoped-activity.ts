type BusySetter = (busy: boolean) => void;
type Entry = { owner: object; pending: number };

/** Busy state belongs to a connection and all its still-running operations. */
export class ScopedActivity {
  private readonly entries = new Map<BusySetter, Entry>();

  begin(owner: object, setBusy: BusySetter): () => void {
    let entry = this.entries.get(setBusy);
    if (!entry || entry.owner !== owner) {
      entry = { owner, pending: 0 };
      this.entries.set(setBusy, entry);
      setBusy(true);
    }
    entry.pending++;
    let finished = false;
    return () => {
      if (finished) return;
      finished = true;
      if (this.entries.get(setBusy) !== entry) return;
      entry.pending--;
      if (entry.pending === 0) {
        this.entries.delete(setBusy);
        setBusy(false);
      }
    };
  }

  reset(): void {
    const setters = [...this.entries.keys()];
    this.entries.clear();
    for (const setBusy of setters) setBusy(false);
  }
}

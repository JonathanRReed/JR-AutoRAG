import { expect, test } from "bun:test";
import { RequestOwner } from "./request-owner";

test("closing an owner stops late body data even if transport ignored cancellation", async () => {
  const owner = new RequestOwner();
  const operation = owner.start();
  let release!: (value: unknown) => void;
  const response = { json: () => new Promise(resolve => { release = resolve; }) } as Response;
  const pending = operation.json(response);
  owner.close();
  release({ stale: true });
  await expect(pending).rejects.toThrow("Connection changed");
  expect(operation.current()).toBe(false);
});

test("strict-effect reactivation never revives cancelled operations", () => {
  const owner = new RequestOwner();
  const old = owner.start();
  owner.close();
  owner.activate();
  const next = owner.start();
  expect(old.current()).toBe(false);
  expect(next.current()).toBe(true);
  next.finish();
  expect(next.current()).toBe(false);
});

test("a newer named read cancels its predecessor without losing cleanup ownership", () => {
  const owner = new RequestOwner();
  const first = owner.start("preview");
  const next = owner.start("preview");
  const independent = owner.start("refresh");
  expect(first.current()).toBe(false);
  first.finish();
  expect(next.current()).toBe(true);
  expect(independent.current()).toBe(true);
  owner.close();
  expect(next.current()).toBe(false);
  expect(independent.current()).toBe(false);
});

import { expect, test } from "bun:test";
import { ScopedActivity } from "./scoped-activity";

test("a late completion cannot clear a newer connection activity", () => {
  const activity = new ScopedActivity();
  const writes: boolean[] = [];
  const setBusy = (value: boolean) => writes.push(value);
  const finishOld = activity.begin({}, setBusy);
  activity.reset();
  const finishNew = activity.begin({}, setBusy);

  finishOld();
  expect(writes).toEqual([true, false, true]);
  finishNew();
  expect(writes).toEqual([true, false, true, false]);
});

test("overlapping operations on the same connection remain busy until all complete", () => {
  const activity = new ScopedActivity();
  const scope = {};
  const writes: boolean[] = [];
  const setBusy = (value: boolean) => writes.push(value);
  const first = activity.begin(scope, setBusy);
  const second = activity.begin(scope, setBusy);

  first();
  first();
  expect(writes).toEqual([true]);
  second();
  expect(writes).toEqual([true, false]);
});

test("replacement and independent controls do not share completion ownership", () => {
  const activity = new ScopedActivity();
  const firstWrites: boolean[] = [];
  const secondWrites: boolean[] = [];
  const first = (value: boolean) => firstWrites.push(value);
  const second = (value: boolean) => secondWrites.push(value);
  const old = activity.begin({}, first);
  const current = activity.begin({}, first);
  const other = activity.begin({}, second);

  old();
  expect(firstWrites).toEqual([true, true]);
  current();
  expect(secondWrites).toEqual([true]);
  other();
  expect(secondWrites).toEqual([true, false]);
});

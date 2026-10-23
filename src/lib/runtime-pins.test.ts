import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

const root = new URL('../../', import.meta.url);
const read = (path: string) => readFileSync(new URL(path, root), 'utf8');
const manifest = JSON.parse(read('package.json'));
const version = manifest.packageManager.replace(/^bun@/, '');

test('Bun package, package manager and CI use the same pinned release', () => {
  expect(version).toMatch(/^\d+\.\d+\.\d+$/);
  expect(manifest.devDependencies.bun).toBe(version);
  expect(read('.github/workflows/ci.yml')).toContain(`BUN_VERSION: "${version}"`);
});

test('the web container uses the same Bun release with an immutable digest', () => {
  const image = read('Dockerfile.web').match(/^FROM oven\/bun:([^\s]+)$/m)?.[1];
  expect(image).toMatch(new RegExp(`^${version.replaceAll('.', '\\.')}-alpine@sha256:[a-f0-9]{64}$`));
});

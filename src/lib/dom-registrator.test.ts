import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';

test('DOM registrator retains isolated globals, safe loading defaults and cleanup', () => {
  const script = `
    import assert from 'node:assert/strict';
    import { GlobalRegistrator } from '@happy-dom/global-registrator';
    GlobalRegistrator.register({ url: 'https://example.test/' });
    assert.equal(window.happyDOM.settings.enableCSSFileLoading, false);
    assert.equal(window.happyDOM.settings.enableIframePageLoading, false);
    const element = document.createElement('div');
    element.style.opacity = '0.5';
    document.body.appendChild(element);
    assert.equal(element.style.opacity, '0.5');
    element.remove();
    assert.equal(document.body.children.length, 0);
    await GlobalRegistrator.unregister();
    assert.equal(typeof document, 'undefined');
    console.log('DOM registration and cleanup passed');
  `;
  const result = spawnSync(process.execPath, ['-e', script], {
    encoding: 'utf8',
    timeout: 10_000,
  });
  expect(result.status, result.stderr || result.error?.message).toBe(0);
  expect(result.stdout).toContain('DOM registration and cleanup passed');
});

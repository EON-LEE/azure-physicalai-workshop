import assert from 'node:assert/strict';
import { readdir, readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const dist = join(root, 'dist');
const manifest = JSON.parse(await readFile(join(dist, '.vite', 'manifest.json'), 'utf8'));
const entries = Object.values(manifest).filter((entry) => entry.isEntry);
assert.equal(entries.length, 1, 'Production must have exactly one entry point');
assert.equal(entries[0].src, 'index.html', 'Only the authenticated production HTML can be an entry point');

const markers = [
  'TEST_ONLY_FACTORY_FIXTURE',
  'test-only-access-token',
  'fixture-operator@example.invalid',
  'test:fixture-server',
  '127.0.0.1:4178',
  '/tests/browser/',
];
let checked = 0;
async function inspect(directory) {
  for (const file of await readdir(directory, { withFileTypes: true })) {
    const path = join(directory, file.name);
    if (file.isDirectory()) await inspect(path);
    else if (/\.(js|css|html|json)$/.test(file.name)) {
      const content = await readFile(path, 'utf8');
      for (const marker of markers) assert(!content.includes(marker), `Test-only content found in ${path}: ${marker}`);
      checked += 1;
    }
  }
}
await inspect(dist);
console.log(`Production boundary verified: one authenticated entry, ${checked} assets, no test-fixture markers.`);

import { readFile, mkdir } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import Ajv2020 from 'ajv/dist/2020.js';
import standaloneCode from 'ajv/dist/standalone/index.js';
import { build } from 'esbuild';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const schema = JSON.parse(await readFile(resolve(root, '..', '..', 'contracts', 'customer-environment.schema.json'), 'utf8'));
const ajv = new Ajv2020({ allErrors: true, strict: true, validateFormats: false, code: { source: true, esm: true } });
const validate = ajv.compile(schema);
const source = `${standaloneCode(ajv, validate)}\nexport const expectedSchema = ${JSON.stringify(schema)};\n`;
const output = join(root, 'src', 'generated');
await mkdir(output, { recursive: true });
await build({
  stdin: { contents: source, resolveDir: root, sourcefile: 'environment-validator.js' },
  outfile: join(output, 'environment-validator.js'),
  bundle: true,
  format: 'esm',
  platform: 'browser',
  target: 'es2022',
  banner: { js: '// Generated from the frozen customer-environment schema. No runtime code generation.' },
  logLevel: 'warning',
});
console.log('Generated CSP-safe standalone environment validator from the frozen contract.');

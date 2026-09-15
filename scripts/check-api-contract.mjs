import { readFileSync } from 'node:fs';
import * as schemas from '../apps/web/src/api/contracts.ts';

const report = JSON.parse(readFileSync(new URL('../test-results/api-contract.json', import.meta.url), 'utf8'));
if (!Array.isArray(report.cases) || report.cases.length === 0) {
  throw new Error('No API response evidence was generated.');
}
for (const item of report.cases) {
  const schema = schemas[item.schema];
  if (!schema || typeof schema.parse !== 'function') {
    throw new Error(`Unknown production web schema: ${item.schema}`);
  }
  schema.parse(item.value);
}
console.log(`${report.cases.length} actual API response shapes accepted by production web decoders.`);
console.log('Test-only dependencies were injected. Azure/GPU execution is NOT verified.');

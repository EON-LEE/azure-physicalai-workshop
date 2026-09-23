import { describe, expect, it } from 'vitest';
import { getEnvironmentValidator, draftFromRecord, expectedRevision, validateDocument } from '../src/environment/validation';
import { environment, environmentSchema, fixtureDocument } from './fixtures/data';

const validator = getEnvironmentValidator(environmentSchema);

describe('customer JSON validation', () => {
  it('accepts the repository reference schema but leaves the source string untouched', () => {
    const raw = ` \n${JSON.stringify(fixtureDocument, null, 4)}\n`;
    const result = validateDocument(raw, validator);
    expect(result.issues).toEqual([]);
    expect(result.document).toEqual(fixtureDocument);
    expect(raw.startsWith(' \n')).toBe(true);
  });

  it('reports duplicate keys without accepting a parse-and-reserialize shadow document', () => {
    const result = validateDocument('{"environment_id":"first","environment_id":"second"}', validator);
    expect(result.document).toBeNull();
    expect(result.issues[0]).toMatchObject({ path: '$.environment_id', message: expect.stringContaining('중복 키') });
  });

  it('finds duplicates inside array objects', () => {
    expect(validateDocument('{"stations":[{"id":"first","id":"second"}]}', validator).issues[0]?.path).toBe('$.stations[0].id');
  });

  it.each(['{', '{"scene": 2,}', '// comment\n{}', '[]', 'null', ''])('rejects nonstandard or non-object JSON: %s', (text) => {
    expect(validateDocument(text, validator).issues.length).toBeGreaterThan(0);
  });

  it('rejects unsupported executable or credential fields using the fetched schema', () => {
    const result = validateDocument(JSON.stringify({ ...fixtureDocument, python: 'print(1)', api_key: 'not-a-real-key' }), validator);
    expect(result.issues).toHaveLength(2);
    expect(result.issues.every((item) => item.message.includes('허용되지 않는 필드'))).toBe(true);
  });

  it('does not declare local validation successful without a server schema', () => {
    expect(validateDocument(JSON.stringify(fixtureDocument), null).issues[0]?.message).toContain('API의 환경 스키마');
  });

  it('fails closed when the API schema differs from the build-time schema', () => {
    expect(() => getEnvironmentValidator({ ...environmentSchema, additionalProperties: true })).toThrow('API 스키마가');
  });

  it('reports excessive nesting instead of crashing the editor', () => {
    const deep = `${'{"child":'.repeat(100)}0${'}'.repeat(100)}`;
    expect(validateDocument(deep, validator).issues[0]?.message).toContain('중첩이 너무 깊습니다');
  });

  it('uses the loaded revision only for that environment ID', () => {
    const draft = draftFromRecord(environment);
    expect(expectedRevision(draft, fixtureDocument)).toBe(environment.revision);
    expect(expectedRevision(draft, { ...fixtureDocument, environment_id: 'new-customer' })).toBeNull();
    expect(expectedRevision({ ...draft, base: null }, fixtureDocument)).toBeNull();
  });

  const learningExecution = {
    schema: 'physicalai.paused-simulation/v1',
    execution_timing: 'paused_simulation',
    profile_id: 'franka-position-hold-10hz-paused-v1',
    max_simulation_seconds: 30,
    max_wall_seconds: 600,
  };
  const learningScene = {
    ...fixtureDocument,
    scene: { ...fixtureDocument.scene, template_id: 'inspection-cell-learning-v1' },
  };
  it('decodes explicit paused-simulation budgets without replacing the legacy wall cap', () => {
    const raw = JSON.stringify({ ...learningScene, learning_execution: learningExecution }, null, 4);
    const result = validateDocument(raw, validator);
    expect(result.issues).toEqual([]);
    expect(result.document?.learning_execution).toEqual(learningExecution);
    expect(result.document?.execution).toEqual(fixtureDocument.execution);
    expect(raw).toContain('"max_wall_seconds": 600');
  });
  it.each([
    { ...learningExecution, max_wall_seconds: 601 },
    { ...learningExecution, max_simulation_seconds: 31 },
    { ...learningExecution, profile_id: 'franka-position-hold-10hz-v1' },
    { ...learningExecution, timing_mode: 'paused_simulation' },
    { ...learningExecution, real_time_admission: true },
    { schema: learningExecution.schema },
  ])('rejects incomplete or unsafe paused-simulation JSON: %j', (learning_execution) => {
    expect(validateDocument(JSON.stringify({ ...learningScene, learning_execution }), validator).issues.length).toBeGreaterThan(0);
  });
  it('does not add a paused default and rejects an opt-in on the legacy reference template', () => {
    expect(validateDocument(JSON.stringify(fixtureDocument), validator).document).not.toHaveProperty('learning_execution');
    expect(validateDocument(JSON.stringify({ ...fixtureDocument, learning_execution: learningExecution }), validator).issues.length).toBeGreaterThan(0);
  });
});

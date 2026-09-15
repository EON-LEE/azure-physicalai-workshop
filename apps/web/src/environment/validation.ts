import type { ErrorObject, ValidateFunction } from 'ajv';
import { getNodeValue, parseTree, printParseErrorCode, type Node, type ParseError } from 'jsonc-parser';
import type { EnvironmentJsonSchema, EnvironmentRecord } from '../api/contracts';
import validateEnvironment, { expectedSchema } from '../generated/environment-validator.js';

export const MAX_DOCUMENT_BYTES = 1024 * 1024;

export interface ValidationIssue {
  path: string;
  message: string;
  offset?: number;
}

export interface DocumentValidation {
  document: Record<string, unknown> | null;
  issues: ValidationIssue[];
}

export interface StudioDraft {
  text: string;
  savedText: string;
  base: EnvironmentRecord | null;
  source: string;
}

export function draftFromRecord(record: EnvironmentRecord): StudioDraft {
  const text = JSON.stringify(record.document, null, 2);
  return { text, savedText: text, base: record, source: '저장된 고객 구성' };
}

export function isDraftDirty(draft: StudioDraft | null): boolean {
  return Boolean(draft && (draft.text !== draft.savedText || (!draft.base && draft.text.length > 0)));
}

function sameSchema(left: unknown, right: unknown): boolean {
  if (left === right) return true;
  if (Array.isArray(left) && Array.isArray(right)) return left.length === right.length && left.every((value, index) => sameSchema(value, right[index]));
  if (!left || !right || typeof left !== 'object' || typeof right !== 'object' || Array.isArray(left) || Array.isArray(right)) return false;
  const entries = Object.entries(left);
  const other = new Map(Object.entries(right));
  return entries.length === other.size && entries.every(([key, value]) => other.has(key) && sameSchema(value, other.get(key)));
}

export function getEnvironmentValidator(schema: EnvironmentJsonSchema): ValidateFunction {
  if (!sameSchema(schema, expectedSchema)) {
    throw new Error('API 스키마가 이 콘솔의 계약과 다릅니다. 콘솔을 다시 빌드하거나 서버 버전을 확인하세요.');
  }
  return validateEnvironment;
}

function duplicateKeys(node: Node, path = '$', depth = 0): ValidationIssue[] {
  if (depth > 64) return [{ path: '$', message: 'JSON 중첩이 너무 깊습니다. 환경 스키마에 맞게 구조를 단순화하세요.' }];
  const issues: ValidationIssue[] = [];
  if (node.type === 'object') {
    const seen = new Set<string>();
    for (const property of node.children ?? []) {
      const key = property.children?.[0];
      const value = property.children?.[1];
      if (!key || typeof key.value !== 'string' || !value) continue;
      if (seen.has(key.value)) {
        issues.push({ path: `${path}.${key.value}`, message: '중복 키입니다. 서버는 중복 JSON 키를 허용하지 않습니다.', offset: key.offset });
      }
      seen.add(key.value);
      issues.push(...duplicateKeys(value, `${path}.${key.value}`, depth + 1));
    }
  } else if (node.type === 'array') {
    for (const [index, child] of (node.children ?? []).entries()) {
      issues.push(...duplicateKeys(child, `${path}[${index}]`, depth + 1));
    }
  }
  return issues;
}

function schemaIssue(error: ErrorObject): ValidationIssue {
  const path = error.instancePath || '$';
  const labels: Record<string, string> = {
    required: '필수 항목이 없습니다',
    additionalProperties: '허용되지 않는 필드입니다',
    type: '값의 자료형을 확인하세요',
    enum: '지원하지 않는 값입니다',
    const: '고정된 값과 다릅니다',
    pattern: '형식이 올바르지 않습니다',
  };
  return { path, message: `${labels[error.keyword] ?? '값의 범위를 확인하세요'}: ${error.message ?? error.keyword} ${JSON.stringify(error.params)}` };
}

export function validateDocument(text: string, validator: ValidateFunction | null): DocumentValidation {
  if (new TextEncoder().encode(text).byteLength > MAX_DOCUMENT_BYTES) {
    return { document: null, issues: [{ path: '$', message: 'JSON 문서는 1 MiB 이하여야 합니다.' }] };
  }
  const errors: ParseError[] = [];
  let root: Node | undefined;
  try {
    root = parseTree(text, errors, { disallowComments: true, allowTrailingComma: false, allowEmptyContent: false });
  } catch {
    return { document: null, issues: [{ path: '$', message: 'JSON의 중첩 구조를 처리할 수 없습니다. 문서 구조를 단순화하세요.' }] };
  }
  if (errors.length || !root) {
    return {
      document: null,
      issues: errors.length ? errors.map((error) => ({
        path: '$', offset: error.offset, message: `JSON 문법 오류: ${printParseErrorCode(error.error)}`,
      })) : [{ path: '$', message: 'JSON 객체를 입력하세요.' }],
    };
  }
  const duplicates = duplicateKeys(root);
  if (duplicates.length) return { document: null, issues: duplicates };
  const parsed: unknown = getNodeValue(root);
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return { document: null, issues: [{ path: '$', message: '최상위 값은 JSON 객체여야 합니다.' }] };
  }
  const document = Object.fromEntries(Object.entries(parsed));
  if (!validator) return { document, issues: [{ path: '$', message: 'API의 환경 스키마를 먼저 불러와야 검사하고 저장할 수 있습니다.' }] };
  if (!validator(document)) return { document, issues: (validator.errors ?? []).map(schemaIssue) };
  return { document, issues: [] };
}

export function expectedRevision(draft: StudioDraft, document: Record<string, unknown>): string | null {
  return draft.base && draft.base.environment_id === document.environment_id ? draft.base.revision : null;
}

export function workflowTarget(environment: EnvironmentRecord, classification: 'accepted' | 'rejected'): string | null {
  const workflow = environment.document.workflow;
  if (typeof workflow !== 'object' || workflow === null) return null;
  const key = classification === 'accepted' ? 'accept_station' : 'reject_station';
  const target = Object.entries(workflow).find(([name]) => name === key)?.[1];
  return typeof target === 'string' ? target : null;
}

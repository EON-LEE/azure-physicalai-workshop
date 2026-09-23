import type { EnvironmentRecord } from '../api/contracts';
import type { TeachingCase } from './contracts';

export function savedTeachingCases(environments: EnvironmentRecord[]): TeachingCase[] {
  return environments.flatMap((record) => {
    const scene = record.document.scene;
    const execution = record.document.execution;
    if (!scene || typeof scene !== 'object' || !('template_id' in scene) || scene.template_id !== 'inspection-cell-learning-v1' ||
      !('seed' in scene) || typeof scene.seed !== 'number' || !Number.isSafeInteger(scene.seed) || scene.seed < 0 || scene.seed === 900002 ||
      !execution || typeof execution !== 'object' || !('mode' in execution) || execution.mode !== 'live' ||
      !('record_demonstration' in execution) || execution.record_demonstration !== true ||
      !('demonstration_split' in execution) || (execution.demonstration_split !== 'train' && execution.demonstration_split !== 'validation')) return [];
    return [{
      case_id: record.environment_id, environment_id: record.environment_id, revision: record.revision,
      seed: scene.seed, split: execution.demonstration_split,
    }];
  });
}

export function defaultTeachingCase(cases: TeachingCase[], environmentId: string, revision: string): string {
  const anchors = cases.filter((item) => item.environment_id === environmentId && item.revision === revision);
  return anchors.length === 1 ? anchors[0]!.case_id : '';
}

export const splitLabel = (split: TeachingCase['split']) => split === 'train' ? '학습 (train)' : '검증 (validation)';

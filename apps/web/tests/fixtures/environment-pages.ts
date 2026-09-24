import type { EnvironmentRecord } from '../../src/api/contracts';
import { environment, fixtureDocument } from './data';

export const environmentCursor = '80000000-1111-4111-8111-111111111111';

export const seventyEnvironments: EnvironmentRecord[] = Array.from({ length: 70 }, (_, index) => {
  const id = `case-${String(index).padStart(3, '0')}`;
  const seed = index < 20 ? 10001 + index : index < 40 ? 11001 + index - 20 : index < 50 ? 20001 + index - 40 : 30001 + index - 50;
  const split = index < 40 ? 'train' : index < 50 ? 'validation' : 'test';
  return {
    ...environment, environment_id: id, display_name: `TEST-ONLY ${id}`,
    revision: (index + 1).toString(16).padStart(64, '0'),
    document: {
      ...fixtureDocument, environment_id: id,
      scene: { ...fixtureDocument.scene, template_id: 'inspection-cell-learning-v1', seed },
      execution: { ...fixtureDocument.execution, record_demonstration: true, demonstration_split: split },
    },
  };
}).reverse();

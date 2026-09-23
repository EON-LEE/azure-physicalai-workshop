import reference from '../../../../examples/inspection-cell.json' with { type: 'json' };

export interface CustomerExperiment {
  experiment: 'quality-gate' | 'relocate-quarantine';
  sample: 'normal' | 'surface_defect';
}

export const defaultExperiment: CustomerExperiment = { experiment: 'quality-gate', sample: 'normal' };
export const inspectionTask = '부품 윗면의 표면 결함을 검사하고, 정상 부품은 정상 트레이로, 결함 부품은 격리 트레이로 보내는 계획을 세워 주세요.';

export function readCustomerExperiment(search: string): { value: CustomerExperiment; error: string | null } {
  const query = new URLSearchParams(search);
  const experiment = query.get('experiment') ?? defaultExperiment.experiment;
  const sample = query.get('sample') ?? defaultExperiment.sample;
  if ((experiment !== 'quality-gate' && experiment !== 'relocate-quarantine') ||
    (sample !== 'normal' && sample !== 'surface_defect')) {
    return { value: defaultExperiment, error: '지원하지 않는 실험 링크입니다. 아래에서 실험과 합성 부품을 다시 선택하세요.' };
  }
  return { value: { experiment, sample }, error: null };
}

export function buildCustomerExperiment(value: CustomerExperiment) {
  const document = structuredClone(reference);
  document.environment_id = `customer-${value.experiment}-${value.sample === 'normal' ? 'normal' : 'defect'}`;
  document.display_name = `${value.experiment === 'quality-gate' ? '검사·분류 기준 실험' : '격리 위치 변경 실험'} · ${value.sample === 'normal' ? '정상 합성 부품' : '결함 합성 부품'}`;
  document.scene.seed = value.sample === 'normal' ? 42 : 43;
  if (value.experiment === 'relocate-quarantine') {
    const quarantine = document.stations.find(station => station.role === 'rejected');
    if (!quarantine) throw new Error('The reference template has no quarantine station.');
    quarantine.position_m = [0.32, -0.38, 0.2];
  }
  return document;
}

export function customerExperimentLink(value: CustomerExperiment) {
  return `/operator?${new URLSearchParams({ view: 'studio', experiment: value.experiment, sample: value.sample })}`;
}

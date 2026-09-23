import { useEffect, useId, useState } from 'react';
import { ArrowUpRight, Download, SlidersHorizontal } from 'lucide-react';
import { buildCustomerExperiment, customerExperimentLink, readCustomerExperiment, type CustomerExperiment } from './customerExperiment';
import './customer-scenario.css';

export function CustomerScenarioPlanner({ onApply }: {
  onApply?: (document: Record<string, unknown>, source: string) => boolean;
}) {
  const [selection, setSelection] = useState(() => readCustomerExperiment(window.location.search));
  const [notice, setNotice] = useState<string | null>(null);
  const id = useId();
  const document = buildCustomerExperiment(selection.value);
  const json = JSON.stringify(document, null, 2);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  useEffect(() => {
    if (selection.error) return;
    const url = URL.createObjectURL(new Blob([json + '\n'], { type: 'application/json' }));
    setDownloadUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [json, selection.error]);
  useEffect(() => {
    const update = () => { setSelection(readCustomerExperiment(window.location.search)); setNotice(null); };
    window.addEventListener('popstate', update);
    return () => window.removeEventListener('popstate', update);
  }, []);

  const select = (value: CustomerExperiment) => {
    setSelection({ value, error: null });
    setNotice(null);
    const url = new URL(window.location.href);
    url.searchParams.set('experiment', value.experiment);
    url.searchParams.set('sample', value.sample);
    window.history.replaceState(null, '', url);
  };

  return <section className="customer-scenario panel" id="customer-lab" aria-labelledby={`${id}-title`}>
    <div className="customer-scenario-heading"><SlidersHorizontal size={21} aria-hidden="true" /><div>
      <p>내 공정에 적용하기 · 로컬 구성 초안</p>
      <h2 id={`${id}-title`}>격리 위치를 바꾸면, 로봇은 어디로 가야 할까요?</h2>
    </div><span>아직 실행하지 않음</span></div>
    <p className="customer-scenario-intro">공장을 먼저 바꾸지 않고 실험할 조건을 정합니다. 아래 선택은 JSON 초안만 바꾸며, 현재 공개 시연이나 GPU를 제어하지 않습니다.</p>
    <div className="customer-scenario-options">
      <fieldset><legend>1. 검증할 고객 질문</legend>
        <label><input type="radio" name={`${id}-experiment`} checked={selection.value.experiment === 'quality-gate'}
          onChange={() => select({ ...selection.value, experiment: 'quality-gate' })} />
          <span><strong>검사 결과에 따라 정상·격리 경로를 나눌 수 있나?</strong><small>기준 셀에서 이미지 판단과 실제 도착 위치를 확인합니다.</small></span></label>
        <label><input type="radio" name={`${id}-experiment`} checked={selection.value.experiment === 'relocate-quarantine'}
          onChange={() => select({ ...selection.value, experiment: 'relocate-quarantine' })} />
          <span><strong>격리 트레이를 10 cm 옮겨도 도달할 수 있나?</strong><small>격리 지점의 X 좌표만 0.22 m에서 0.32 m로 변경합니다.</small></span></label>
      </fieldset>
      <fieldset><legend>2. 관측할 합성 부품</legend>
        <label><input type="radio" name={`${id}-sample`} checked={selection.value.sample === 'normal'}
          onChange={() => select({ ...selection.value, sample: 'normal' })} />
          <span><strong>정상 표면</strong><small>정상 트레이로 계획하는지 확인합니다.</small></span></label>
        <label><input type="radio" name={`${id}-sample`} checked={selection.value.sample === 'surface_defect'}
          onChange={() => select({ ...selection.value, sample: 'surface_defect' })} />
          <span><strong>표면 결함 표식</strong><small>격리 트레이로 계획하는지 확인합니다.</small></span></label>
      </fieldset>
    </div>
    <div className="customer-scenario-expectation">
      <strong>실행 후 확인할 것 · 아래는 예측 성공 결과가 아닙니다</strong>
      <p>{selection.value.experiment === 'relocate-quarantine'
        ? '환경을 저장·활성화한 뒤, 격리 대상 계획의 좌표와 실제 최종 위치를 비교하세요. JSON 저장이나 좌표 변경만으로 물리 도달을 보장하지 않습니다.'
        : '실제 검사 원본과 분류 이유를 검토하고, 승인한 대상 트레이에 부품이 도착했는지 측정값으로 확인하세요.'}</p>
      <p>Foundry에는 선택한 정답이 아니라 실제 카메라 이미지를 보냅니다. 잘못된 계획은 승인하지 마세요.</p>
    </div>
    {selection.error && <p className="customer-scenario-error" role="alert">{selection.error}</p>}
    <details><summary>생성된 환경 JSON 확인</summary><pre aria-label="생성된 고객 실험 JSON" translate="no">{json}</pre></details>
    <div className="customer-scenario-actions">
      {downloadUrl && !selection.error && <a className="customer-action secondary" href={downloadUrl} download={`${document.environment_id}.json`}><Download size={16} aria-hidden="true" />환경 JSON 다운로드</a>}
      {onApply ? <button type="button" className="customer-action" disabled={Boolean(selection.error)}
        onClick={() => { if (onApply(document, '고객 공정 실험 초안')) setNotice('새 초안을 편집기에 넣었습니다. 저장·씬 활성화·계획 검토·승인은 각각 직접 진행하세요.'); }}>
        이 구성으로 새 초안 만들기
      </button> : !selection.error && <a className="customer-action" href={customerExperimentLink(selection.value)}>운영자에게 이 실험 전달<ArrowUpRight size={16} aria-hidden="true" /></a>}
    </div>
    {notice && <p role="status" className="customer-scenario-notice">{notice}</p>}
    <p className="customer-scenario-boundary">다운로드는 로그인 없이 가능합니다. 실행은 권한 있는 운영자가 로그인한 뒤 환경 저장 → 씬 활성화 → 관측·계획 → 승인 → 결과 확인 순으로 진행합니다. 공개 시연과 같은 GPU를 쓰므로 운영자가 사용 시간을 조율해야 합니다.</p>
  </section>;
}

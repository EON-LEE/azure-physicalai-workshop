import { beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { CustomerScenarioPlanner } from '../src/environment/CustomerScenarioPlanner';
import { buildCustomerExperiment, customerExperimentLink, readCustomerExperiment } from '../src/environment/customerExperiment';
import { getEnvironmentValidator } from '../src/environment/validation';
import { ConsoleApp } from '../src/ConsoleApp';
import { account, environmentSchema } from './fixtures/data';
import { makeApi } from './helpers';

beforeEach(() => window.history.replaceState(null, '', '/'));

describe('customer experiment authoring, not pretend robot controls', () => {
  it.each(['normal', 'surface_defect'] as const)('changes only customer identity, fixture sample and quarantine X for %s', sample => {
    const original = buildCustomerExperiment({ experiment: 'quality-gate', sample });
    const moved = buildCustomerExperiment({ experiment: 'relocate-quarantine', sample });
    expect(getEnvironmentValidator(environmentSchema)(moved)).toBe(true);
    expect(moved.scene.seed).toBe(sample === 'normal' ? 42 : 43);
    expect(moved.stations.find(station => station.id === 'rejected')?.position_m).toEqual([0.32, -0.38, 0.2]);
    expect(original.stations.find(station => station.id === 'rejected')?.position_m).toEqual([0.22, -0.38, 0.2]);
    expect(moved.stations.filter(station => station.id !== 'rejected')).toEqual(original.stations.filter(station => station.id !== 'rejected'));
    expect(moved.limits).toEqual(original.limits);
    expect(moved.execution).toEqual(original.execution);
    expect(moved.workflow).toEqual(original.workflow);
  });

  it('changes a real downloadable configuration and exact operator link without calling any API', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch');
    render(<CustomerScenarioPlanner />);
    await userEvent.click(screen.getByRole('radio', { name: /격리 트레이를 10 cm/ }));
    await userEvent.click(screen.getByRole('radio', { name: /표면 결함 표식/ }));
    const selection = { experiment: 'relocate-quarantine', sample: 'surface_defect' } as const;
    expect(screen.getByRole('link', { name: '운영자에게 이 실험 전달' })).toHaveAttribute('href', customerExperimentLink(selection));
    expect(JSON.parse(screen.getByLabelText('생성된 고객 실험 JSON').textContent ?? '')).toEqual(buildCustomerExperiment(selection));
    expect(screen.getByRole('link', { name: '환경 JSON 다운로드' })).toHaveAttribute('download', 'customer-relocate-quarantine-defect.json');
    expect(window.location.search).toContain('experiment=relocate-quarantine');
    expect(window.location.search).toContain('sample=surface_defect');
    expect(fetch).not.toHaveBeenCalled();
    expect(screen.getByText('아직 실행하지 않음')).toBeInTheDocument();
    expect(URL.revokeObjectURL).toHaveBeenCalled();
  });

  it('surfaces invalid handoff links and cannot generate an active action from them', async () => {
    window.history.replaceState(null, '', '/?experiment=execute-code&sample=unknown');
    expect(readCustomerExperiment(window.location.search).error).not.toBeNull();
    render(<CustomerScenarioPlanner />);
    expect(screen.getByRole('alert')).toHaveTextContent('지원하지 않는 실험 링크');
    expect(screen.queryByRole('link', { name: '운영자에게 이 실험 전달' })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('radio', { name: /격리 트레이를 10 cm/ }));
    expect(screen.getByRole('link', { name: '운영자에게 이 실험 전달' })).toBeInTheDocument();
  });

  it('opens the operator editor with the exact unsaved experiment, never saving, activating or approving automatically', async () => {
    const selection = { experiment: 'relocate-quarantine', sample: 'surface_defect' } as const;
    window.history.replaceState(null, '', customerExperimentLink(selection));
    const api = makeApi();
    render(<ConsoleApp api={api} account={account} />);
    const editor = await screen.findByLabelText('고객 환경 JSON 원본');
    expect(editor).toHaveValue(JSON.stringify(buildCustomerExperiment(selection), null, 2));
    await waitFor(() => expect(screen.getByRole('button', { name: 'JSON 저장' })).toBeEnabled());
    for (const operation of [api.saveEnvironment, api.activateEnvironment, api.createRun, api.approveRun]) expect(operation).not.toHaveBeenCalled();
    expect(api.getFrame).not.toHaveBeenCalled();
    fireEvent.change(editor, { target: { value: '{ "my": "unsaved draft" }' } });
    vi.spyOn(window, 'confirm').mockReturnValue(false);
    await userEvent.click(screen.getByRole('button', { name: '이 구성으로 새 초안 만들기' }));
    expect(editor).toHaveValue('{ "my": "unsaved draft" }');
  });
});

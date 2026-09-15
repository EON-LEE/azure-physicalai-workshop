import { useState } from 'react';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import type { ConsoleApi, EnvironmentRecord } from '../src/api/contracts';
import { ApiError } from '../src/api/errors';
import { draftFromRecord, type StudioDraft } from '../src/environment/validation';
import { EnvironmentStudio } from '../src/views/EnvironmentStudio';
import { environment, fixtureDocument } from './fixtures/data';
import { makeApi } from './helpers';

function StudioHarness({ api, onSaved = vi.fn(), onActivate = vi.fn(), initial = environment }: {
  api: ConsoleApi; onSaved?: (record: EnvironmentRecord) => void; onActivate?: (record: EnvironmentRecord) => void; initial?: EnvironmentRecord;
}) {
  const [draft, setDraft] = useState<StudioDraft | null>(draftFromRecord(initial));
  return <EnvironmentStudio api={api} environments={[initial]} initialEnvironment={initial} draft={draft} setDraft={setDraft}
    onSaved={onSaved} runtime={null} runtimeError={null} runtimeFresh={false}
    activation={{ submitting: false, receipt: null, error: null, timedOut: false }} onActivate={onActivate} onRefresh={vi.fn()} />;
}

describe('revision-aware Environment Studio', () => {
  it('saves raw edited JSON and the loaded revision without reformatting or activating', async () => {
    const api = makeApi({ saveEnvironment: vi.fn().mockResolvedValue({ ...environment, revision: 'b'.repeat(64) }) });
    const onActivate = vi.fn();
    render(<StudioHarness api={api} onActivate={onActivate} />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'JSON 저장' })).toBeEnabled());
    const raw = ` \n${JSON.stringify({ ...fixtureDocument, display_name: '고객 편집 원본' }, null, 4)}\n\n`;
    fireEvent.change(screen.getByLabelText('고객 환경 JSON 원본'), { target: { value: raw } });
    await userEvent.click(screen.getByRole('button', { name: 'JSON 저장' }));
    await waitFor(() => expect(api.saveEnvironment).toHaveBeenCalledWith(raw, environment.revision, expect.any(AbortSignal)));
    expect(screen.getByLabelText('고객 환경 JSON 원본')).toHaveValue(raw);
    expect(await screen.findByText(/원본 JSON을 저장했습니다/)).toBeInTheDocument();
    expect(onActivate).not.toHaveBeenCalled();
    expect(screen.queryByText('요청한 버전의 런타임 준비 확인됨')).not.toBeInTheDocument();
  });

  it('retains imported CRLF bytes in document_json and the explicitly loaded base revision', async () => {
    vi.spyOn(window, 'confirm').mockReturnValue(true);
    const api = makeApi({ saveEnvironment: vi.fn().mockResolvedValue(environment) });
    render(<StudioHarness api={api} />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'JSON 저장' })).toBeEnabled());
    const raw = ` \r\n${JSON.stringify(fixtureDocument, null, 2).replace(/\n/g, '\r\n')}\r\n `;
    const file = new File([raw], 'customer.json', { type: 'application/json' });
    Object.defineProperty(file, 'text', { value: async () => raw });
    await userEvent.upload(screen.getByLabelText('환경 JSON 파일'), file);
    await screen.findByText('가져온 JSON: customer.json');
    await userEvent.click(screen.getByRole('button', { name: 'JSON 저장' }));
    expect(api.saveEnvironment).toHaveBeenCalledWith(raw, environment.revision, expect.any(AbortSignal));
  });

  it('preserves edits and loaded revision after 409 without retrying an unconditional write', async () => {
    const api = makeApi({ saveEnvironment: vi.fn().mockRejectedValue(new ApiError('revision_conflict', '다른 사용자가 환경을 수정했습니다.', 409)) });
    const confirm = vi.spyOn(window, 'confirm').mockReturnValue(false);
    render(<StudioHarness api={api} />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'JSON 저장' })).toBeEnabled());
    const raw = JSON.stringify({ ...fixtureDocument, display_name: '수정 내용을 보존하세요' }, null, 4);
    fireEvent.change(screen.getByLabelText('고객 환경 JSON 원본'), { target: { value: raw } });
    await userEvent.click(screen.getByRole('button', { name: 'JSON 저장' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('저장 버전 충돌');
    expect(screen.getByLabelText('고객 환경 JSON 원본')).toHaveValue(raw);
    expect(api.saveEnvironment).toHaveBeenCalledTimes(1);
    expect(api.saveEnvironment).toHaveBeenCalledWith(raw, environment.revision, expect.any(AbortSignal));
    await userEvent.click(screen.getByRole('button', { name: /최신 버전 불러오기/ }));
    expect(confirm).toHaveBeenCalled();
    expect(api.getEnvironments).not.toHaveBeenCalled();
    expect(screen.getByLabelText('고객 환경 JSON 원본')).toHaveValue(raw);
  });

  it('surfaces server semantic validation details while retaining the raw document', async () => {
    const api = makeApi({ saveEnvironment: vi.fn().mockRejectedValue(new ApiError('invalid_environment', '스테이션이 작업 영역 밖에 있습니다.', 422, [{ path: '$.stations[0]' }])) });
    render(<StudioHarness api={api} />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'JSON 저장' })).toBeEnabled());
    const raw = JSON.stringify(fixtureDocument, null, 4);
    fireEvent.change(screen.getByLabelText('고객 환경 JSON 원본'), { target: { value: raw } });
    await userEvent.click(screen.getByRole('button', { name: 'JSON 저장' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('스테이션이 작업 영역 밖에 있습니다.');
    expect(screen.getByLabelText('고객 환경 JSON 원본')).toHaveValue(raw);
  });

  it('does not drop duplicate keys through client parsing', async () => {
    const api = makeApi();
    render(<StudioHarness api={api} />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'JSON 검사' })).toBeEnabled());
    const raw = '{"environment_id":"one", "environment_id":"two"}';
    fireEvent.change(screen.getByLabelText('고객 환경 JSON 원본'), { target: { value: raw } });
    await userEvent.click(screen.getByRole('button', { name: 'JSON 저장' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('중복 키');
    expect(api.saveEnvironment).not.toHaveBeenCalled();
    expect(screen.getByLabelText('고객 환경 JSON 원본')).toHaveValue(raw);
  });

  it('disables save if the server schema is missing or incompatible', async () => {
    const api = makeApi({ getEnvironmentSchema: vi.fn().mockResolvedValue({ type: 'object', additionalProperties: true }) });
    render(<StudioHarness api={api} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('API 스키마가 이 콘솔의 계약과 다릅니다');
    expect(screen.getByRole('button', { name: 'JSON 저장' })).toBeDisabled();
  });

  it('allows REPLAY JSON editing but never exposes a replay activation or fake live control', async () => {
    const replay = { ...environment, document: { ...fixtureDocument, execution: { ...fixtureDocument.execution, mode: 'replay' } } };
    render(<StudioHarness api={makeApi()} initial={replay} />);
    await waitFor(() => expect(screen.getByRole('button', { name: 'JSON 저장' })).toBeEnabled());
    expect(screen.getByText('REPLAY 구성')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '저장된 씬 활성화' })).not.toBeInTheDocument();
    expect(screen.getByText('현재 미제공')).toBeInTheDocument();
  });
});

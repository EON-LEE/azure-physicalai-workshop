import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { ReferenceCollections } from '../src/learning/ReferenceCollections';
import { learningApi } from './fixtures/learning';
import { referenceFixture } from './fixtures/reference-collection';
const project = () => referenceFixture().project;

const stages = {
  execution_timing: 'paused_simulation' as const, real_time_admission: false as const,
  supported: true as const, enabled: false, status: 'reference_only', message: 'TEST ONLY',
  reference_generation_enabled: true, training_enabled: false, evaluation_enabled: false, release_enabled: false,
};

describe('explicit reference generation, not manual NRT teaching', () => {
  it('requires deliberate approval and sends only the selected immutable case/request', async () => {
    const data = project();
    const api = learningApi();
    api.startReference.mockRejectedValue(new Error('TEST ONLY missing installed runtime grant'));
    render(<ReferenceCollections api={api} project={data} stages={stages} />);
    const start = screen.getByRole('button', { name: '승인한 REFERENCE 시연 생성' });
    expect(start).toBeDisabled();
    await userEvent.selectOptions(screen.getByLabelText('기준 시연 배치'), data.item.teaching_cases[0]!.case_id);
    await userEvent.click(screen.getByRole('checkbox', { name: '선택한 배치의 검토된 기준 제어기 이동을 승인합니다' }));
    await userEvent.click(start);
    await waitFor(() => expect(api.startReference).toHaveBeenCalledTimes(1));
    expect(api.startReference.mock.calls[0]?.[1]).toEqual({
      request_id: expect.any(String), case_id: data.item.teaching_cases[0]!.case_id, motion_approved: true,
    });
    expect(api.teach).not.toHaveBeenCalled();
    expect(api.jog).not.toHaveBeenCalled();
    expect(api.train).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: 'X 양의 방향 5mm' })).not.toBeInTheDocument();
  });

  it('does not make reference permission authorize training or evaluation', () => {
    const api = learningApi();
    render(<ReferenceCollections api={api} project={project()} stages={stages} />);
    expect(screen.getByRole('button', { name: '검증된 데이터로 제한된 학습 제출' })).toBeDisabled();
    expect(screen.getByRole('button', { name: '전체 고정 조건의 제한된 후보 평가' })).toBeDisabled();
    expect(screen.getByText(/직접 손으로 가르치는 NRT 조작은 아직 제공하지 않습니다/)).toBeInTheDocument();
  });

  it('keeps the reference button blocked when the operator has not admitted the stage', () => {
    render(<ReferenceCollections api={learningApi()} project={project()} stages={{ ...stages, reference_generation_enabled: false }} />);
    expect(screen.getByRole('button', { name: '승인한 REFERENCE 시연 생성' })).toBeDisabled();
  });

  it('recovers the original collection and reports actual WALL/SIM values without a new start', async () => {
    const fixture = referenceFixture();
    const api = learningApi();
    api.records.mockResolvedValue({ items: [fixture.collection] });
    api.reference.mockResolvedValue(fixture.collection);
    render(<ReferenceCollections api={api} project={fixture.project} stages={stages} />);
    await userEvent.click(await screen.findByText('기존 기준 시연 기록 · 원래 명령 조회'));
    await userEvent.click(screen.getByRole('button', { name: '기준 시연 기록 보기' }));
    expect(await screen.findByText('REFERENCE · running · 캡처 pending')).toBeInTheDocument();
    expect(screen.getByText('2000')).toBeInTheDocument();
    expect(screen.getByText('0.1')).toBeInTheDocument();
    expect(api.startReference).not.toHaveBeenCalled();
    expect(api.reference).toHaveBeenCalledWith(fixture.collection.item.id, expect.any(AbortSignal));
    expect(screen.getByRole('button', { name: '승인한 REFERENCE 시연 생성' })).toBeDisabled();
  });

  it('keeps uncertain or running cancellation distinct from confirmed termination', async () => {
    const fixture = referenceFixture();
    const api = learningApi();
    const cancelling = { ...fixture.collection, item: { ...fixture.collection.item, status: 'cancelling' as const } };
    api.records.mockResolvedValue({ items: [fixture.collection] });
    api.reference.mockResolvedValueOnce(fixture.collection).mockResolvedValue(cancelling);
    api.cancelReference.mockResolvedValue(cancelling);
    render(<ReferenceCollections api={api} project={fixture.project} stages={stages} />);
    await userEvent.click(await screen.findByText('기존 기준 시연 기록 · 원래 명령 조회'));
    await userEvent.click(screen.getByRole('button', { name: '기준 시연 기록 보기' }));
    await userEvent.click(await screen.findByRole('button', { name: '기준 시연 취소 요청' }));
    expect(await screen.findByText('REFERENCE · cancelling · 캡처 pending')).toBeInTheDocument();
    expect(api.cancelReference).toHaveBeenCalledTimes(1);
    expect(screen.getByRole('button', { name: '기준 시연 취소 요청' })).toBeDisabled();
  });
});

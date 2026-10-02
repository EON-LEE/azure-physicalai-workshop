import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { evaluationSchema } from '../src/learning/contracts';
import { LearningJobPanel } from '../src/learning/LearningJobPanel';
import { commandModelAdmissionSchema, simulationReportSchema } from '../src/learning/simulationReports';
import { learningApi, learningFixture } from './fixtures/learning';
import { simulationReportFixture } from './fixtures/simulation-report';

function mappedReport() {
  const { results_sha256: _oldResults, ...native } = simulationReportFixture();
  return {
    ...native, native_schema: 'physicalai.managed-paired-report/v1',
    mapping_sha256: 'a'.repeat(64), evidence_sha256: 'b'.repeat(64),
    trials: native.trials.map((trial, index) => {
      const physical = `00000000-0000-4000-8000-${String(index + 1).padStart(12, '0')}`;
      return {
        ...trial, episode_id: physical, physical_attempt_id: physical,
        logical_case_id: `logical-${trial.seed}`,
      };
    }),
  };
}

describe('operator-bound managed evaluation imports', () => {
  it('retains only the explicit separately pinned command-model admission shape', () => {
    const original = mappedReport();
    const admission = {
      admission_kind: 'azureml_command_v3', artifact_schema: 'physicalai.smolvla-checkpoint/v3',
      training_execution: 'azureml_command', server_entrypoint: 'learning.paused.command_model',
      provider_entrypoint: 'simulation.command_policy_deployment.CommandPausedPolicyProvider',
      request_schema: 'physicalai.smolvla-request/v2', response_schema: 'physicalai.smolvla-response/v2',
      runtime_sha256: '1'.repeat(64), legacy_servo_sha256: '2'.repeat(64),
      control_profile_sha256: original.control_profile_sha256,
      simulator_sources_sha256: '3'.repeat(64), native_sources_sha256: '4'.repeat(64),
      simulator_image: `test.azurecr.io/test@sha256:${'5'.repeat(64)}`,
      simulator_source_revision: '6'.repeat(40),
    };
    expect(commandModelAdmissionSchema.parse(admission)).toEqual(admission);
    expect(simulationReportSchema.parse({ ...original, model_admission: admission }).model_admission).toEqual(admission);
    expect(simulationReportSchema.parse(original)).not.toHaveProperty('model_admission');
    expect(() => simulationReportSchema.parse({
      ...original, model_admission: { ...admission, control_profile_sha256: '7'.repeat(64) },
    })).toThrow();
    expect(() => simulationReportSchema.parse({
      ...simulationReportFixture(), model_admission: admission,
    })).toThrow();
    expect(() => commandModelAdmissionSchema.parse({
      ...admission, server_entrypoint: 'learning.paused.model',
    })).toThrow();
  });

  it('retains all physical and logical IDs without inventing an old results.json', () => {
    const report = simulationReportSchema.parse(mappedReport());
    expect(report.trials).toHaveLength(40);
    expect(report).not.toHaveProperty('results_sha256');
    expect(report.trials[0]?.physical_attempt_id).toBe(report.trials[0]?.episode_id);
    expect(() => simulationReportSchema.parse({ ...mappedReport(), results_sha256: 'c'.repeat(64) })).toThrow();
    expect(() => simulationReportSchema.parse({ ...mappedReport(), trials: mappedReport().trials.slice(1) })).toThrow();
    const duplicate = mappedReport();
    duplicate.trials[1] = duplicate.trials[0]!;
    expect(() => simulationReportSchema.parse(duplicate)).toThrow();
  });

  it('shows pending managed imports without an Azure ML ID, paid retry or physical cancel control', async () => {
    const fixture = learningFixture().evaluation;
    const record = {
      ...fixture.item, provider: 'managed_batch', status: 'awaiting_import', azure_job_id: null,
      azure_status: null, backend_status: null, report: null,
      execution_timing: 'paused_simulation', real_time_admission: false,
      control_profile_id: 'franka-position-hold-10hz-paused-v1',
      control_profile_sha256: 'a'.repeat(64), criteria_sha256: 'b'.repeat(64), frozen_plan_sha256: 'c'.repeat(64),
    };
    const item = evaluationSchema.parse(record);
    const api = learningApi();
    api.job.mockResolvedValue({ ...fixture, item });
    render(<LearningJobPanel api={api} initial={{ ...fixture, item }} />);
    expect(await screen.findByText('managed_batch · 검증된 자료 가져오기')).toBeInTheDocument();
    expect(screen.getByText('완전한 평가 자료 가져오기 대기')).toBeInTheDocument();
    expect(screen.queryByText('Azure ML job ID')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '실제 작업 취소 요청' })).not.toBeInTheDocument();
    expect(() => evaluationSchema.parse({ ...record, azure_job_id: '/invented/job' })).toThrow();
    expect(() => evaluationSchema.parse({ ...record, status: 'succeeded', report: mappedReport() })).toThrow();
    expect(api.train).not.toHaveBeenCalled();
    expect(api.evaluate).not.toHaveBeenCalled();
  });
});

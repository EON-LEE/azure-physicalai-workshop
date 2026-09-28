import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { candidateSchema, externalTrainingImportSchema } from '../src/learning/contracts';
import { TeachingStudio } from '../src/learning/TeachingStudio';
import { learningApi, learningFixture } from './fixtures/learning';
import { referenceFixture } from './fixtures/reference-collection';

function legacyCandidate() {
  const { project, dataset, training, evaluation } = learningFixture();
  return candidateSchema.parse({
    id: evaluation.item.candidate_id, actor_id: project.item.actor_id,
    created_at: project.item.created_at, updated_at: project.item.updated_at,
    kind: 'candidate', project_id: project.item.id, dataset_id: dataset.item.id,
    training_run_id: training.item.id, parent_release_id: project.item.baseline_release_id,
    pretrained_artifact_id: null, policy_type: 'smolvla',
    model_sha256: 'b'.repeat(64), parent_model_sha256: 'a'.repeat(64),
    processor_sha256: 'c'.repeat(64), manifest_sha256: 'b'.repeat(64),
    artifact_id: evaluation.item.candidate_id, optimizer_steps: 100,
    azure_job_id: training.item.azure_job_id, source_commit: 'd'.repeat(40),
    model_revision: 'e'.repeat(40), control_profile_id: project.item.control_profile_id,
  });
}

function importedRecord() {
  const { project } = referenceFixture();
  const old = legacyCandidate();
  const now = '2026-09-28T04:30:00Z';
  const job = 'dddddddd-dddd-4ddd-8ddd-dddddddddddd';
  const checksum = 'a'.repeat(64);
  return externalTrainingImportSchema.parse({
    kind: 'external_import', schema: 'physicalai.external-training-import/v1',
    id: 'eeeeeeee-eeee-4eee-8eee-eeeeeeeeeeee', actor_id: old.actor_id,
    project_id: project.item.id, created_at: now, updated_at: now, imported_at: now,
    native_job_name: job, azure_job_id: `/subscriptions/test/workspaces/test/jobs/${job}`,
    azure_job_type: 'command', native_created_at: '2026-09-28T03:47:45Z',
    job_deadline_utc: '2026-09-28T04:43:54Z', approval_sha256: checksum,
    plan_sha256: checksum, plan_archive_sha256: checksum, configuration_sha256: checksum,
    specification_sha256: checksum, image_qualification_sha256: checksum,
    environment_image: `test.azurecr.io/test@sha256:${checksum}`,
    managed_identity_client_id: old.actor_id, code_snapshot_sha256: checksum,
    static_source_sha256: checksum, completion_sha256: checksum, result_sha256: checksum,
    transfer_sha256: checksum, model_sha256: old.model_sha256, parent_model_sha256: old.parent_model_sha256,
    backbone_sha256: checksum, raw_manifest_sha256: checksum, candidate_id: old.id,
    dataset_id: old.dataset_id, optimizer_steps: 1000, learning_quality_verified: false,
    execution_timing: 'paused_simulation', real_time_admission: false,
    control_profile_id: project.item.control_profile_id,
    control_profile_sha256: project.item.control_profile_sha256,
    criteria_sha256: project.item.criteria_sha256, frozen_plan_sha256: project.item.frozen_plan_sha256,
  });
}

describe('private external-native training provenance', () => {
  it('keeps external import exclusive with API training and does not upgrade legacy records', () => {
    const old = legacyCandidate();
    const imported = importedRecord();
    const value = {
      ...old, policy_type: 'smolvla', training_origin: 'external_native_import',
      training_run_id: null, external_import_id: imported.id,
      parent_release_id: null, pretrained_artifact_id: null,
      execution_timing: imported.execution_timing, real_time_admission: false,
      control_profile_id: imported.control_profile_id, control_profile_sha256: imported.control_profile_sha256,
      criteria_sha256: imported.criteria_sha256, frozen_plan_sha256: imported.frozen_plan_sha256,
    };
    expect(candidateSchema.parse(value).training_run_id).toBeNull();
    expect(candidateSchema.parse(old)).not.toHaveProperty('training_origin');
    expect(() => candidateSchema.parse({ ...value, training_run_id: old.training_run_id })).toThrow();
    expect(() => candidateSchema.parse({ ...value, external_import_id: undefined })).toThrow();
    expect(() => externalTrainingImportSchema.parse({ ...imported, learning_quality_verified: true })).toThrow();
  });

  it('shows actual post-hoc records privately without submitting, publishing or inventing an API run', async () => {
    const api = learningApi();
    const { project } = referenceFixture();
    const imported = importedRecord();
    api.projects.mockResolvedValue({ items: [project] });
    api.records.mockImplementation(async (_project, kind) => ({
      items: kind === 'external_import' ? [{ item: imported, etag: 'test-only' }] : [],
    }));
    window.history.replaceState(null, '', `/operator?view=learning&learning_project=${project.item.id}`);
    render(<TeachingStudio api={api} environments={[]} />);
    expect(await screen.findByRole('heading', { name: '외부에서 학습됨 · 검증 후 가져옴' })).toBeInTheDocument();
    expect(screen.getByText(imported.azure_job_id)).toBeInTheDocument();
    expect(screen.getByText(/사후 검증한 기록/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '실제 작업 취소 요청' })).not.toBeInTheDocument();
    expect(api.train).not.toHaveBeenCalled();
    expect(api.release).not.toHaveBeenCalled();
    expect(api.job).not.toHaveBeenCalled();
  });
});

import type { RunRecord } from '../../src/api/contracts';
import { environment, pendingRun } from './data';

export const skillReleaseId = '60000000-1111-4111-8111-111111111111';
export const skillPlanId = '70000000-1111-4111-8111-111111111111';
export const skillInstruction = 'Pick up the synthetic part from the source platform and place it in the quarantine tray.';
export const skillRun = {
  ...pendingRun,
  id: 'test-only-released-skill-run',
  execution_mode: 'released_skill',
  instruction: skillInstruction,
  policy: {
    policy_release_id: skillReleaseId, policy_type: 'smolvla',
    model_sha256: 'c'.repeat(64), processor_sha256: 'd'.repeat(64), manifest_sha256: 'c'.repeat(64),
    control_profile_id: 'franka-position-hold-10hz-v1', task_id: 'manufacturing-part-placement-v1',
    goal_station_id: 'rejected', instruction: skillInstruction,
  },
  plan: {
    kind: 'released_skill', skill_plan_id: skillPlanId, policy_release_id: skillReleaseId,
    policy_type: 'smolvla', model_sha256: 'c'.repeat(64),
    task_id: 'manufacturing-part-placement-v1', instruction: skillInstruction,
    target_station_id: 'rejected', object_id: 'part-001',
    summary: 'TEST-ONLY: approved released task; no CV inspection or Foundry call.',
    observation_id: 'test-only-skill-observation', epoch: 'test-only-world-epoch', state_revision: 1,
    expires_at: new Date(Date.now() + 60_000).toISOString(),
  },
  environment_id: environment.environment_id,
} satisfies RunRecord;

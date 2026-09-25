import type { SimulationReport } from '../../src/learning/simulationReports';

export function simulationReportFixture(): SimulationReport {
  const wall = { samples: 2, p50: 250, p95: 260, max: 260 };
  return {
    execution_timing: 'paused_simulation', real_time_admission: false,
    native_schema: 'physicalai.smolvla-paired-report/v2', comparison_kind: 'paired_policy',
    control_profile_id: 'franka-position-hold-10hz-paused-v1', control_profile_sha256: 'a'.repeat(64),
    criteria_sha256: 'b'.repeat(64), frozen_plan_sha256: 'c'.repeat(64),
    evaluation_plan_sha256: 'd'.repeat(64), native_plan_sha256: 'e'.repeat(64),
    results_sha256: 'f'.repeat(64), runtime_sha256: '1'.repeat(64),
    report_sha256: '2'.repeat(64), artifact_id: '20000000-1111-4111-8111-111111111111',
    before_model_sha256: '3'.repeat(64), after_model_sha256: '4'.repeat(64),
    candidate_model_sha256: null, reference_controller_sha256: null,
    counts: { before: { total: 20, success: 17 }, after: { total: 20, success: 18 } },
    success_rates: { before: .85, after: .9 }, absolute_success_rate_improvement: .05,
    latency_wall_ms: { before: { ...wall, samples: 40 }, after: { ...wall, samples: 40 } },
    total_wall_duration_ms: 56000, total_simulation_duration_ms: 8000,
    safety_violation_count: 0, resource_violation_count: 0, total_trial_count: 40,
    quality_gate_passed: true, conclusion: 'improved', live_gpu_verified: true,
    trials: Array.from({ length: 40 }, (_, index) => {
      const policy = index < 20 ? 'before' : 'after';
      const success = index % 20 < (policy === 'before' ? 17 : 18);
      return {
        episode_id: `test-only-trial-${index % 20}`, seed: 30001 + index % 20, attempt: 0, policy,
        model_sha256: (policy === 'before' ? '3' : '4').repeat(64),
        environment_id: `test-only-case-${index % 20}`, revision: '5'.repeat(64),
        observed_initial_pose_m: [.3, 0, .1], scene_builder_sha256: '6'.repeat(64),
        final_pose_m: [.5, .2, .1], destination_id: 'rejected', terminated: true, truncated: false,
        failure_reason: success ? null : 'test-only-unsettled', safety_violation_count: 0,
        policy_predict_calls: 2, applied_action_count: 12, reference_route_calls: 0,
        final_images: { inspection: '7'.repeat(64), overview: '8'.repeat(64) },
        phase_wall_ms: { policy: wall, observation: wall, hold: wall, interval: { ...wall, max: 600 }, heartbeat: { ...wall, samples: 12 } },
        wall_duration_ms: 1400, simulation_duration_ms: 200, physical_success: success,
        position_error_m: [0, 0, 0],
        task_evidence: {
          predicate_version: 'physicalai.measured-grasp-transport/v1',
          predicate_source: { commit: 'a'.repeat(40), path: 'simulation/control.py', git_blob: 'b'.repeat(40), sha256: 'c'.repeat(64), symbol: 'TaskWatchdog' },
          grasp_evidence_kind: 'measured_lift_proximity_finger_gap_no_contact_sensor',
          grasp_verified: success, settled: success, settled_simulation_seconds: success ? .3 : 0,
          final_goal_error_m: 0, maximum_tcp_speed_m_s: .1, safety_violation_count: 0,
        },
      };
    }),
  };
}

import type { JogBody, LearningApi, LearningRecord, Resource } from '../../src/learning/contracts';
import { learningFixture } from '../fixtures/learning-data';
import { referenceFixture } from '../fixtures/reference-collection';

export interface LearningTrace { calls: string[]; inputs: JogBody[]; referenceRequests?: Parameters<LearningApi['startReference']>[1][]; }
export function browserLearning(trace: LearningTrace, enabled: boolean, referenceOnly = false): LearningApi {
  const data = learningFixture();
  const reference = referenceFixture();
  let referenceStarted = false;
  let session = data.teaching;
  const mark = (name: string) => trace.calls.push(name);
  const unsupported = async (): Promise<never> => { throw new Error('TEST ONLY: this operation has no injected response.'); };
  return {
    async capabilities() { mark('capabilities'); return {
      enabled, status: enabled ? 'configured' : 'disabled', message: 'TEST ONLY: no actual model or GPU verification.',
      policy_types: enabled ? ['smolvla'] : [], control_profiles: ['franka-position-hold-10hz-v1'],
      training_verified: false, coach_configured: false, bootstrap_allowed: false,
      simulation_learning: {
        execution_timing: 'paused_simulation', real_time_admission: false,
        supported: true, enabled: false, status: 'producer_verifier_unavailable',
        message: 'TEST ONLY: the separate paused runtime and report integration is blocked.',
        reference_generation_enabled: referenceOnly, training_enabled: false, evaluation_enabled: false, release_enabled: false,
      },
    }; },
    async projects() { mark('projects'); return { items: [referenceOnly ? reference.project : data.project] }; },
    createProject: unsupported,
    async records(_id, kind) {
      const all: Resource<LearningRecord>[] = [data.dataset, data.training, data.evaluation];
      if (referenceOnly) return { items: kind === 'reference_collection' && referenceStarted ? [reference.collection] : [] };
      return { items: all.filter((entry) => entry.item.kind === kind) };
    },
    async teach(_id, body) {
      mark('teach');
      const selectedCase = data.project.item.teaching_cases.find((item) => item.case_id === body.case_id);
      if (!selectedCase) throw new Error('TEST ONLY: only an approved teaching case may start.');
      session = { ...session, item: { ...session.item, source: body.source, teaching_case: selectedCase } };
      return session;
    },
    async teaching() { return session; },
    async startReference(_id, body) {
      if (!referenceOnly) return unsupported();
      mark('reference'); (trace.referenceRequests ??= []).push(body);
      referenceStarted = true;
      return reference.collection;
    },
    async reference() {
      if (!referenceOnly || !referenceStarted) return unsupported();
      return reference.collection;
    },
    cancelReference: unsupported,
    async arm(_id, body) { mark('arm'); return { ...data.grant, item: { ...data.grant.item, sequence: body.sequence, delta_xyz_m: body.delta_xyz_m, gripper: body.gripper } }; },
    async jog(_id, body) {
      mark('jog'); trace.inputs.push(body);
      session = { ...session, item: { ...session.item, last_sequence: body.sequence } };
      return session;
    },
    async teachingControl(_id, action) {
      mark(action); session = { ...session, item: { ...session.item, status: action === 'finish' ? 'uploading' : 'cancelling', physical_status: action === 'finish' ? 'succeeded' : 'cancelling' } };
      return session;
    },
    seal: unsupported,
    artifactOperation: unsupported,
    dataset: async () => data.dataset,
    async train() { mark('train'); return data.training; },
    async evaluate() { mark('evaluate'); return data.evaluation; },
    async job(id) { return id === data.evaluation.item.id ? data.evaluation : data.training; },
    reportDocument: unsupported,
    cancelJob: unsupported,
    release: unsupported,
    coach: unsupported,
  };
}

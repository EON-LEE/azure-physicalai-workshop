import { vi, type Mocked } from 'vitest';
import type { LearningApi } from '../../src/learning/contracts';
import { learningFixture } from './learning-data';
export { learningFixture } from './learning-data';

export function learningApi(): Mocked<LearningApi> {
  const fixture = learningFixture();
  const unexpected = async (): Promise<never> => { throw new Error('Unexpected test-only learning operation; inject a response explicitly.'); };
  return {
    capabilities: vi.fn<LearningApi['capabilities']>().mockResolvedValue({
      enabled: true, status: 'configured', message: 'TEST ONLY', policy_types: ['smolvla'],
      control_profiles: ['franka-position-hold-10hz-v1'], training_verified: false, coach_configured: false, bootstrap_allowed: false,
    }),
    projects: vi.fn<LearningApi['projects']>().mockResolvedValue({ items: [fixture.project] }),
    createProject: vi.fn<LearningApi['createProject']>().mockImplementation(unexpected),
    records: vi.fn<LearningApi['records']>().mockResolvedValue({ items: [] }),
    teach: vi.fn<LearningApi['teach']>().mockImplementation(unexpected),
    teaching: vi.fn<LearningApi['teaching']>().mockResolvedValue(fixture.teaching),
    arm: vi.fn<LearningApi['arm']>().mockResolvedValue(fixture.grant),
    jog: vi.fn<LearningApi['jog']>().mockImplementation(async (_id, body) => ({ ...fixture.teaching, item: { ...fixture.teaching.item, last_sequence: body.sequence } })),
    teachingControl: vi.fn<LearningApi['teachingControl']>().mockImplementation(unexpected),
    seal: vi.fn<LearningApi['seal']>().mockImplementation(unexpected),
    train: vi.fn<LearningApi['train']>().mockImplementation(unexpected),
    evaluate: vi.fn<LearningApi['evaluate']>().mockImplementation(unexpected),
    job: vi.fn<LearningApi['job']>().mockResolvedValue(fixture.training),
    reportDocument: vi.fn<LearningApi['reportDocument']>().mockImplementation(unexpected),
    cancelJob: vi.fn<LearningApi['cancelJob']>().mockImplementation(unexpected),
    release: vi.fn<LearningApi['release']>().mockImplementation(unexpected),
    coach: vi.fn<LearningApi['coach']>().mockImplementation(unexpected),
  };
}

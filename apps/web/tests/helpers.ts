import { vi, type Mocked } from 'vitest';
import type { ConsoleApi } from '../src/api/contracts';
import { environment, environmentSchema, fixtureDocument, fixturePng, runtime } from './fixtures/data';

function unexpected(): never {
  throw new Error('Unexpected mutation in test. Configure this API response explicitly.');
}

export function makeApi(overrides: Partial<Mocked<ConsoleApi>> = {}): Mocked<ConsoleApi> {
  return {
    getRuntime: vi.fn<ConsoleApi['getRuntime']>().mockResolvedValue(runtime),
    getEnvironments: vi.fn<ConsoleApi['getEnvironments']>().mockResolvedValue({ items: [environment] }),
    getEnvironmentSchema: vi.fn<ConsoleApi['getEnvironmentSchema']>().mockResolvedValue(environmentSchema),
    getTemplates: vi.fn<ConsoleApi['getTemplates']>().mockResolvedValue({ items: [{ name: '참조 검사 셀', document: fixtureDocument }] }),
    saveEnvironment: vi.fn<ConsoleApi['saveEnvironment']>().mockImplementation(unexpected),
    activateEnvironment: vi.fn<ConsoleApi['activateEnvironment']>().mockImplementation(unexpected),
    getFrame: vi.fn<ConsoleApi['getFrame']>().mockImplementation(async () => ({ blob: fixturePng(), frameId: 'test-only-frame-id', capturedAt: new Date().toISOString(), physicsSteps: 125 })),
    getRuns: vi.fn<ConsoleApi['getRuns']>().mockResolvedValue({ items: [] }),
    getRun: vi.fn<ConsoleApi['getRun']>().mockImplementation(unexpected),
    createRun: vi.fn<ConsoleApi['createRun']>().mockImplementation(unexpected),
    approveRun: vi.fn<ConsoleApi['approveRun']>().mockImplementation(unexpected),
    cancelRun: vi.fn<ConsoleApi['cancelRun']>().mockImplementation(unexpected),
    getObservation: vi.fn<ConsoleApi['getObservation']>().mockResolvedValue(fixturePng()),
    ...overrides,
  };
}

export function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (error: unknown) => void;
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

export function setVisibility(state: 'visible' | 'hidden') {
  Object.defineProperty(document, 'visibilityState', { configurable: true, value: state });
  document.dispatchEvent(new Event('visibilitychange'));
}

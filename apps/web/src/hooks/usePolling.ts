import { useCallback, useEffect, useRef, useState } from 'react';
import { canRetryRead, isAbort } from '../api/errors';
import { usePageVisible } from './usePageVisible';

interface PollingOptions<T> {
  active?: boolean;
  intervalMs?: number | null;
  initialData?: T | null;
  continuePolling?: (data: T) => boolean;
}

export function usePolling<T>(
  load: (signal: AbortSignal) => Promise<T>,
  { active = true, intervalMs = null, initialData = null, continuePolling }: PollingOptions<T> = {},
) {
  const visible = usePageVisible();
  const [data, setData] = useState<T | null>(initialData);
  const [error, setError] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const [lastReceivedAt, setLastReceivedAt] = useState<number | null>(null);
  const [generation, setGeneration] = useState(0);
  const lastStartedAt = useRef<number | null>(null);
  const refresh = useCallback(() => setGeneration((value) => value + 1), []);

  useEffect(() => {
    if (!active || !visible) {
      setLoading(false);
      return;
    }
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let controller: AbortController | undefined;
    let failures = 0;
    const poll = async () => {
      lastStartedAt.current = Date.now();
      controller = new AbortController();
      setLoading(true);
      try {
        const result = await load(controller.signal);
        if (!alive) return;
        failures = 0;
        setData(result);
        setError(null);
        setLastReceivedAt(Date.now());
        if (intervalMs !== null && (!continuePolling || continuePolling(result))) {
          timer = setTimeout(poll, Math.max(1000, intervalMs));
        }
      } catch (failure) {
        if (!alive || isAbort(failure)) return;
        setError(failure);
        if (intervalMs !== null && canRetryRead(failure)) {
          failures += 1;
          timer = setTimeout(poll, Math.min(30_000, Math.max(1000, intervalMs) * 2 ** Math.min(failures, 5)));
        }
      } finally {
        if (alive) setLoading(false);
      }
    };
    const cooldown = intervalMs !== null && lastStartedAt.current !== null ? Math.max(0, 1000 - (Date.now() - lastStartedAt.current)) : 0;
    if (cooldown) timer = setTimeout(poll, cooldown);
    else void poll();
    return () => {
      alive = false;
      clearTimeout(timer);
      controller?.abort();
    };
  }, [load, active, visible, intervalMs, continuePolling, generation]);

  return { data, setData, error, loading, lastReceivedAt, refresh, paused: !active || !visible };
}

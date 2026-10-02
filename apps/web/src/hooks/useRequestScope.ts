import { useCallback, useEffect, useRef } from 'react';

export function useRequestScope() {
  const controllers = useRef(new Set<AbortController>());
  const mounted = useRef(false);
  useEffect(() => {
    mounted.current = true;
    const pending = controllers.current;
    return () => {
      mounted.current = false;
      for (const controller of pending) controller.abort();
      pending.clear();
    };
  }, []);

  const start = useCallback(() => {
    const controller = new AbortController();
    controllers.current.add(controller);
    return {
      signal: controller.signal,
      current: () => mounted.current && !controller.signal.aborted,
      finish: () => controllers.current.delete(controller),
    };
  }, []);
  return start;
}

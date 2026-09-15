import { useEffect, useState } from 'react';
import { usePolling } from './usePolling';

export function useProtectedImage<T extends { blob: Blob }>(
  load: (signal: AbortSignal) => Promise<T>,
  intervalMs: number | null,
  active = true,
) {
  const resource = usePolling(load, { active, intervalMs });
  const [url, setUrl] = useState<string | null>(null);
  const [clock, setClock] = useState(Date.now);
  const [decodeError, setDecodeError] = useState(false);

  useEffect(() => {
    if (resource.paused || !resource.data) {
      setUrl(null);
      return;
    }
    const nextUrl = URL.createObjectURL(resource.data.blob);
    setUrl(nextUrl);
    setDecodeError(false);
    return () => URL.revokeObjectURL(nextUrl);
  }, [resource.data, resource.paused]);

  useEffect(() => {
    if (resource.paused || intervalMs === null) return;
    setClock(Date.now());
    const timer = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [resource.paused, intervalMs]);

  return { ...resource, url, clock, decodeError, onDecodeError: () => setDecodeError(true) };
}

import { useCallback, useEffect, useRef, useState } from 'react';
import type { ConsoleApi, EnvironmentRecord, Environments } from '../api/contracts';
import { ApiError, isAbort } from '../api/errors';
import { usePageVisible } from './usePageVisible';
import { usePolling } from './usePolling';

interface Cache {
  scope: string;
  api: ConsoleApi;
  items: EnvironmentRecord[];
  nextCursor: string | null;
  conflicts: string[];
}

export function useEnvironmentPages(api: ConsoleApi, scope: string, active: boolean) {
  const [cache, setCache] = useState<Cache>({ api, scope, items: [], nextCursor: null, conflicts: [] });
  const [pageError, setPageError] = useState<unknown>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const more = useRef<AbortController | null>(null);
  const generation = useRef(0);
  const visible = usePageVisible();
  const load = useCallback(async (signal: AbortSignal) => {
    const sequence = ++generation.current;
    more.current?.abort();
    more.current = null;
    setLoadingMore(false);
    setPageError(null);
    const page = await api.getEnvironments(signal);
    return { api, scope, sequence, page };
  }, [api, scope]);
  const first = usePolling(load, { active });
  const merge = useCallback((page: Environments) => {
    setCache((previous) => {
      const sameScope = previous.api === api && previous.scope === scope;
      const records = new Map((sameScope ? previous.items : []).map((item) => [item.environment_id, item]));
      const conflicts = new Set(sameScope ? previous.conflicts : []);
      for (const item of page.items) {
        const prior = records.get(item.environment_id);
        if (prior && prior.revision !== item.revision) conflicts.add(item.environment_id);
        records.set(item.environment_id, item);
      }
      return { api, scope, items: [...records.values()], nextCursor: page.next_cursor ?? null, conflicts: [...conflicts] };
    });
  }, [api, scope]);
  useEffect(() => {
    const result = first.data;
    if (result?.api === api && result.scope === scope && result.sequence === generation.current) {
      merge(result.page);
      setPageError(null);
    }
  }, [first.data, api, scope, merge]);
  useEffect(() => {
    if (!active || !visible) {
      more.current?.abort();
      more.current = null;
      setLoadingMore(false);
    }
    return () => {
      more.current?.abort();
      more.current = null;
    };
  }, [active, visible, api, scope]);
  const inScope = cache.api === api && cache.scope === scope;
  const loadMore = async () => {
    if (!inScope || !active || !visible || !cache.nextCursor || first.loading || more.current) return;
    const controller = new AbortController();
    more.current = controller;
    const sequence = generation.current;
    setLoadingMore(true);
    setPageError(null);
    try {
      const page = await api.getEnvironments(controller.signal, cache.nextCursor);
      if (controller.signal.aborted || sequence !== generation.current) return;
      merge(page);
    } catch (error) {
      if (!controller.signal.aborted && sequence === generation.current && !isAbort(error)) setPageError(error);
    } finally {
      if (more.current === controller) {
        more.current = null;
        setLoadingMore(false);
      }
    }
  };
  const upsert = useCallback((record: EnvironmentRecord) => {
    setCache((value) => {
      const sameScope = value.api === api && value.scope === scope;
      const records = sameScope ? value.items.filter((item) => item.environment_id !== record.environment_id) : [];
      return { api, scope, items: [...records, record], nextCursor: sameScope ? value.nextCursor : null, conflicts: sameScope ? value.conflicts : [] };
    });
  }, [api, scope]);
  const refresh = () => {
    setCache((value) => ({ ...value, conflicts: [] }));
    setPageError(null);
    first.refresh();
  };
  const conflict = inScope && cache.conflicts.length
    ? new ApiError('environment_revision_changed', `같은 환경의 다른 저장 버전이 도착했습니다: ${cache.conflicts.slice(0, 3).join(', ')}. 기존 선택을 검토하고 변경된 항목은 다시 선택하세요.`, 409)
    : null;
  return {
    items: inScope ? cache.items : [], nextCursor: inScope ? cache.nextCursor : null,
    error: pageError ?? first.error ?? conflict, loading: first.loading, loadingMore,
    loadMore, refresh, upsert, paused: first.paused,
  };
}

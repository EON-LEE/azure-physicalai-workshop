import { lazy, Suspense } from 'react';
import { DemoViewer } from './public/DemoViewer';

const OperatorEntry = lazy(() => import('./auth/OperatorEntry'));

export function AppEntry() {
  const operator = window.location.pathname === '/operator' || window.location.pathname.startsWith('/operator/');
  return operator
    ? <Suspense fallback={<main className="demo-load" aria-live="polite">운영자 화면을 불러오는 중…</main>}><OperatorEntry /></Suspense>
    : <DemoViewer />;
}

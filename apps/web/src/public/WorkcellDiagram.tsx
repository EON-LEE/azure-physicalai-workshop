import { useId } from 'react';
import type { DemoStation } from './api';

const labels = { source: '부품 투입', inspection: '비전 검사', accepted: '다음 공정', rejected: '불량 격리' };
function point(station: DemoStation): [number, number] {
  return [400 + station.position_m[0] * 260 + station.position_m[1] * 140,
    288 - station.position_m[1] * 150 + station.position_m[0] * 42];
}

export function WorkcellDiagram({ stations, path, step }: {
  stations: DemoStation[];
  path: 'accepted' | 'rejected';
  step: number;
}) {
  const id = useId().replaceAll(':', '');
  const source = stations.find((item) => item.role === 'source');
  const inspection = stations.find((item) => item.role === 'inspection');
  const destination = stations.find((item) => item.role === path);
  const route = [source, inspection, destination].filter((item): item is DemoStation => Boolean(item)).map(point);
  const role = step === 0 ? 'source' : step < 3 ? 'inspection' : path;
  return (
    <svg className="demo-cell-svg" viewBox="0 0 800 490" role="img" aria-labelledby={`${id}-title ${id}-desc`}>
      <title id={`${id}-title`}>검사·분류 작업 셀 구성도</title>
      <desc id={`${id}-desc`}>실제 시뮬레이션 영상이 아닌 구성도입니다. 부품 투입, 비전 검사, 다음 공정, 불량 격리 위치를 표시합니다.</desc>
      <defs>
        <pattern id={`${id}-grid`} width="40" height="40" patternUnits="userSpaceOnUse">
          <path d="M 40 0 L 0 0 0 40" fill="none" stroke="#263d57" strokeWidth="0.6" />
        </pattern>
        <linearGradient id={`${id}-floor`} x1="0" y1="0" x2="1" y2="1">
          <stop stopColor="#233d5d" /><stop offset="1" stopColor="#162b44" />
        </linearGradient>
        <linearGradient id={`${id}-arm`} x1="0" y1="0" x2="1" y2="1">
          <stop stopColor="#edf5ff" /><stop offset="1" stopColor="#8fa8c5" />
        </linearGradient>
      </defs>
      <rect width="800" height="490" fill="#101f32" />
      <rect width="800" height="490" fill={`url(#${id}-grid)`} />
      <path d="M120 319 365 144 725 270 484 446Z" fill={`url(#${id}-floor)`} stroke="#3b5471" />
      <path d="M120 319v13l364 128 241-178v-12L484 446Z" fill="#0c1829" stroke="#29415d" />
      <polyline points={route.map(([x, y]) => `${x},${y}`).join(' ')} fill="none"
        stroke={path === 'rejected' ? '#f6b85a' : '#5fe0c2'} strokeWidth="3" strokeDasharray="8 8" />
      {stations.map((station) => {
        const [x, y] = point(station);
        const active = station.role === role;
        const color = station.role === 'rejected' ? '#f6b85a' : station.role === 'accepted' ? '#5fe0c2' : '#69b6ff';
        return (
          <g key={station.id}>
            <ellipse cx={x} cy={y + 16} rx="56" ry="24" fill="#0a1525" opacity=".7" />
            <path d={`M${x - 43} ${y - 2}l43-24 49 18-45 28Z`} fill={active ? color : '#314c6b'} fillOpacity={active ? .8 : 1} stroke={color} strokeWidth={active ? 2 : 1} />
            <path d={`M${x - 43} ${y - 2}v17l47 26v-21Zm47 22v21l45-28v-23Z`} fill="#1c324b" stroke="#54718e" />
            {station.role === 'source' && <g fill="#80d1c0" stroke="#c8f9eb">
              <path d={`M${x - 18} ${y - 18}l15-8 19 8-15 9Z`} />
              <path d={`M${x - 18} ${y - 18}v14l19 9V${y - 9}Zm19 9v14l15-9v-14Z`} fill="#3b8b80" />
            </g>}
            {station.role === 'inspection' && <g stroke="#72bdff" fill="#214263">
              <path d={`M${x - 27} ${y - 17}v-57h54v57`} strokeWidth="5" fill="none" />
              <rect x={x - 12} y={y - 84} width="24" height="18" rx="3" />
              <path d={`M${x - 8} ${y - 66}l-23 52h62l-23-52Z`} fill="#69b6ff" fillOpacity=".12" strokeDasharray="3 4" />
            </g>}
            <g transform={`translate(${x - 55} ${y + 43})`}>
              <rect width="110" height="34" rx="7" fill="#14283e" stroke={active ? color : '#36526f'} />
              <circle cx="14" cy="17" r="4" fill={color} />
              <text x="27" y="22" fill="#e6effa" fontSize="13" fontWeight="600">{labels[station.role]}</text>
            </g>
          </g>
        );
      })}
      <g aria-hidden="true">
        <ellipse cx="394" cy="296" rx="43" ry="18" fill="#091625" opacity=".75" />
        <path d="m368 280 27-14 28 15-27 16Z" fill="#c3d2e3" />
        <path d="M368 280v16l28 16v-15l27-16v15l-27 16" fill="#7c96b5" />
        <path d="M394 280v-50l-40-51 23-48 54 13 15 48" fill="none" stroke="#071423" strokeWidth="29" strokeLinecap="round" strokeLinejoin="round" />
        <path d="M394 280v-50l-40-51 23-48 54 13 15 48" fill="none" stroke={`url(#${id}-arm)`} strokeWidth="20" strokeLinecap="round" strokeLinejoin="round" />
        {[[394, 230], [354, 179], [377, 131], [431, 144]].map(([x, y]) => <g key={`${x}-${y}`}>
          <circle cx={x} cy={y} r="13" fill="#e0edfc" stroke="#7894b5" strokeWidth="3" />
          <circle cx={x} cy={y} r="5" fill="#2889e7" />
        </g>)}
        <path d="m440 190-3 17m16-18 5 15m-21 3 6 8m15-11-4 9" stroke="#e0edfc" strokeWidth="5" fill="none" />
      </g>
      <g fill="#92a8c3" fontSize="11" letterSpacing="2">
        <text x="31" y="42">REFERENCE WORKCELL / 01</text>
        <text x="31" y="466">SCENE SCHEMATIC · NOT A LIVE SIMULATION</text>
      </g>
      <g transform="translate(588 30)">
        <rect width="183" height="29" rx="14" fill="#20394e" />
        <text x="15" y="19" fill="#a7cee8" fontSize="11">Franka · Synthetic scene</text>
      </g>
    </svg>
  );
}

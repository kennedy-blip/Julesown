'use client';

import React from 'react';
import { readinessOf, useHealth } from '@/lib/health';
import type { Readiness } from '@/lib/health';

const BADGES: Record<Readiness, { label: string; className: string; dot: string; title: string }> = {
  loading: {
    label: 'Checking backend…',
    className: 'bg-slate-500/10 text-slate-400 border-slate-500/20',
    dot: 'bg-slate-400',
    title: 'Asking the backend what is configured',
  },
  ready: {
    label: 'Backend ready',
    className: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    dot: 'bg-emerald-400',
    title: 'An LLM key, the sandbox key and GitHub write credentials are configured',
  },
  incomplete: {
    label: 'Setup incomplete',
    className: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    dot: 'bg-amber-400',
    title: 'The backend is missing configuration needed to run tasks',
  },
  locked: {
    label: 'Locked',
    className: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    dot: 'bg-amber-400',
    title: 'Task execution is locked until JULES_ACCESS_KEY is set on the deployment',
  },
  unreachable: {
    label: 'Backend unreachable',
    className: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
    dot: 'bg-rose-400',
    title: 'The API did not answer',
  },
};

/** Header badge reflecting what the backend reports about itself. */
export function SystemStatus() {
  const readiness = readinessOf(useHealth());
  const badge = BADGES[readiness];
  return (
    <span
      className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium border whitespace-nowrap ${badge.className}`}
      title={badge.title}
      data-testid="system-status"
      data-readiness={readiness}
    >
      <span className={`w-2 h-2 rounded-full mr-1.5 ${badge.dot} ${readiness === 'ready' ? 'animate-pulse' : ''}`}></span>
      {badge.label}
    </span>
  );
}

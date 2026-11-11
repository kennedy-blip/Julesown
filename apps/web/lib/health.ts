'use client';

import { useEffect, useSyncExternalStore } from 'react';
import { describeError, fetchHealth } from './api';
import type { HealthEnv } from './api';

/**
 * Backend readiness, shared by the header badge and the dashboard so both show
 * the same thing from a single request. Nothing here is assumed: until the
 * backend has answered, the state is "loading", and if it cannot be reached
 * the state says so.
 */
export type HealthState =
  | { phase: 'loading' }
  | { phase: 'ok'; data: HealthEnv }
  | { phase: 'unreachable'; message: string };

export type Readiness = 'loading' | 'ready' | 'incomplete' | 'locked' | 'unreachable';

const LOADING: HealthState = { phase: 'loading' };

let current: HealthState = LOADING;
let inflight: Promise<void> | null = null;
let requested = false;
const listeners: Array<() => void> = [];

function publish(next: HealthState): void {
  current = next;
  listeners.slice().forEach((listener) => listener());
}

export function refreshHealth(): Promise<void> {
  if (inflight) return inflight;
  requested = true;
  inflight = fetchHealth()
    .then((data) => publish({ phase: 'ok', data }))
    .catch((error) => publish({ phase: 'unreachable', message: describeError(error) }))
    .then(() => {
      inflight = null;
    });
  return inflight;
}

function subscribe(listener: () => void): () => void {
  listeners.push(listener);
  return () => {
    const index = listeners.indexOf(listener);
    if (index >= 0) listeners.splice(index, 1);
  };
}

export function useHealth(): HealthState {
  const state = useSyncExternalStore(
    subscribe,
    () => current,
    () => LOADING
  );
  useEffect(() => {
    if (!requested) void refreshHealth();
  }, []);
  return state;
}

export function readinessOf(state: HealthState): Readiness {
  if (state.phase === 'loading') return 'loading';
  if (state.phase === 'unreachable') return 'unreachable';
  const mode = state.data.access ? state.data.access.mode : 'open';
  if (mode === 'locked') return 'locked';
  const ready = typeof state.data.ready === 'boolean' ? state.data.ready : state.data.status === 'HEALTHY';
  return ready ? 'ready' : 'incomplete';
}

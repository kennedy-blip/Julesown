'use client';

import React, { useEffect, useRef, useState } from 'react';
import { Terminal as TerminalIcon, Copy, Check } from 'lucide-react';
import type { LogEntry } from '@/lib/api';
import type { SessionState } from '@/lib/sessions';
import { Linkified } from './Linkified';

interface LiveTerminalProps {
  logs: LogEntry[];
  state: SessionState | 'idle';
  /** Where the commands run, e.g. "E2B sandbox abc123" (reported by the backend). */
  sandbox?: string | null;
  startedAt?: number | null; // ms
  finishedAt?: number | null; // ms
}

const LEVEL_STYLES: Record<string, string> = {
  client: 'text-slate-500',
  info: 'text-slate-300',
  stage: 'text-blue-400 font-semibold',
  tool: 'text-slate-400',
  edit: 'text-amber-300',
  command: 'text-cyan-300',
  output: 'text-slate-500',
  agent: 'text-violet-300',
  success: 'text-emerald-400',
  warn: 'text-amber-400',
  error: 'text-rose-400',
};

const LEVEL_PREFIX: Record<string, string> = {
  stage: '▸ ',
  output: '  ',
  warn: '! ',
  error: '✗ ',
};

function formatClock(ts: number): string {
  const date = new Date(ts * 1000);
  const pad = (value: number) => value.toString().padStart(2, '0');
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
}

function formatElapsed(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${seconds.toString().padStart(2, '0')}`;
}

export function LiveTerminal({ logs, state, sandbox, startedAt, finishedAt }: LiveTerminalProps) {
  const bodyRef = useRef<HTMLDivElement>(null);
  const stickToBottom = useRef(true);
  const [copied, setCopied] = useState(false);
  const [now, setNow] = useState(() => Date.now());

  // Follow new output, unless the user has scrolled up to read something.
  useEffect(() => {
    const body = bodyRef.current;
    if (body && stickToBottom.current) body.scrollTop = body.scrollHeight;
  }, [logs]);

  useEffect(() => {
    if (state !== 'running') return;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [state]);

  const onScroll = () => {
    const body = bodyRef.current;
    if (!body) return;
    stickToBottom.current = body.scrollHeight - body.scrollTop - body.clientHeight < 40;
  };

  const copyLogs = async () => {
    const text = logs.map((entry) => `[${formatClock(entry.ts)}] ${entry.message}`).join('\n');
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch {
      /* clipboard unavailable (insecure context or permission denied) */
    }
  };

  const elapsed =
    startedAt && state === 'running'
      ? formatElapsed(now - startedAt)
      : startedAt && finishedAt
        ? formatElapsed(finishedAt - startedAt)
        : null;

  const BADGES: Record<SessionState | 'idle', { label: string; className: string; dot: string }> = {
    running: { label: 'Running', className: 'text-amber-300 bg-amber-500/10 border-amber-500/25', dot: 'bg-amber-300 animate-ping' },
    completed: { label: 'Finished', className: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/25', dot: 'bg-emerald-400' },
    failed: { label: 'Failed', className: 'text-rose-400 bg-rose-500/10 border-rose-500/25', dot: 'bg-rose-400' },
    interrupted: { label: 'Connection lost', className: 'text-amber-400 bg-amber-500/10 border-amber-500/25', dot: 'bg-amber-400' },
    idle: { label: 'Idle', className: 'text-slate-400 bg-slate-700/40 border-slate-600/50', dot: 'bg-slate-500' },
  };
  const badge = () => {
    const style = BADGES[state] || BADGES.idle;
    return (
      <span
        className={`inline-flex items-center gap-1.5 h-6 px-2 rounded-md border text-[11px] font-medium ${style.className}`}
        data-testid="terminal-state"
        data-state={state}
      >
        <span className={`w-1.5 h-1.5 rounded-full ${style.dot}`}></span>
        {style.label}
      </span>
    );
  };

  return (
    <section className="bg-[#0f172a] border border-slate-800 rounded-xl overflow-hidden shadow-lg shadow-black/20 flex flex-col h-[400px]">
      <header className="px-5 h-12 flex-shrink-0 border-b border-slate-800 flex items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 min-w-0 text-[13px] font-semibold text-slate-100">
          <TerminalIcon className="w-4 h-4 text-slate-400 flex-shrink-0" />
          <span className="flex-shrink-0">Activity</span>
          <span className="text-xs font-mono font-normal text-slate-500 truncate" data-testid="terminal-title">
            {sandbox || 'nothing running'}
          </span>
        </h2>

        <div className="flex items-center gap-3 flex-shrink-0">
          {elapsed && <span className="text-xs font-mono text-slate-400 tabular-nums" data-testid="elapsed">{elapsed}</span>}
          {badge()}
          <button
            onClick={copyLogs}
            disabled={logs.length === 0}
            className="text-slate-400 hover:text-slate-200 transition-colors p-1.5 rounded-md hover:bg-slate-800 disabled:opacity-40"
            title="Copy log"
            aria-label="Copy log"
          >
            {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
          </button>
        </div>
      </header>

      <div
        ref={bodyRef}
        onScroll={onScroll}
        className="px-4 py-3 flex-1 overflow-y-auto space-y-0.5 bg-[#050914] font-mono"
        data-testid="terminal-body"
      >
        {logs.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-500 text-xs italic text-center px-6">
            Nothing has run yet. When you start a task, every step the agent takes in the sandbox is
            streamed here as it happens.
          </div>
        ) : (
          logs.map((entry, index) => (
            <div
              key={index}
              className={`font-mono text-xs leading-5 tracking-tight whitespace-pre-wrap break-words hover:bg-slate-800/40 px-1 rounded ${
                LEVEL_STYLES[entry.level] || 'text-slate-300'
              }`}
            >
              <span className="text-slate-600 select-none mr-2">{formatClock(entry.ts)}</span>
              {LEVEL_PREFIX[entry.level] || ''}
              <Linkified text={entry.message} />
            </div>
          ))
        )}
      </div>
    </section>
  );
}

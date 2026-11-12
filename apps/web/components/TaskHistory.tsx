'use client';

import React from 'react';
import { History, Plus, Trash2 } from 'lucide-react';
import { sessionBadge } from '@/lib/sessions';
import type { Session, SessionBadge } from '@/lib/sessions';
import { Badge, Button, Panel } from './ui';
import type { BadgeTone } from './ui';

const BADGES: Record<SessionBadge, { label: string; tone: BadgeTone }> = {
  RUNNING: { label: 'Running', tone: 'warn' },
  'PR OPENED': { label: 'PR opened', tone: 'good' },
  'PR UPDATED': { label: 'PR updated', tone: 'good' },
  UNFINISHED: { label: 'Unfinished', tone: 'warn' },
  MERGED: { label: 'Merged', tone: 'accent' },
  'PR CLOSED': { label: 'PR closed', tone: 'muted' },
  PUSHED: { label: 'Pushed', tone: 'info' },
  'NO CHANGES': { label: 'No changes', tone: 'muted' },
  FAILED: { label: 'Failed', tone: 'bad' },
  UNKNOWN: { label: 'Unknown', tone: 'warn' },
};

function formatWhen(ms: number): string {
  const date = new Date(ms);
  const sameDay = new Date().toDateString() === date.toDateString();
  const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return sameDay ? time : `${date.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`;
}

/** What a task was about, in one line. */
export function describeSession(session: Session): string {
  const target = session.target || null;
  const prompt = session.prompt.trim();
  if (target && target.fixChecks) {
    const where = target.pullRequest ? `pull request #${target.pullRequest}` : 'the branch';
    return `Fix the failed checks of ${where}${prompt ? `: ${prompt}` : ''}`;
  }
  if (prompt) return prompt;
  if (target && target.continueWork) return 'Continue the unfinished work';
  return session.task && session.task.recovered ? 'Recovered from GitHub' : '—';
}

interface TaskHistoryProps {
  sessions: Session[];
  activeId: string | null;
  onSelect: (session: Session) => void;
  onNew: () => void;
  onClear: () => void;
}

/** The tasks run from this browser, newest first. */
export function TaskHistory({ sessions, activeId, onSelect, onNew, onClear }: TaskHistoryProps) {
  const finished = sessions.some((session) => session.state !== 'running');
  return (
    <Panel
      icon={<History className="w-4 h-4 text-slate-400" />}
      title="Tasks"
      aside={
        <>
          {finished && (
            <button
              type="button"
              onClick={onClear}
              className="p-1.5 rounded-md text-slate-400 hover:text-rose-300 hover:bg-slate-800 transition-colors"
              title="Remove finished tasks from this browser's history"
              aria-label="Clear the task history"
              data-testid="history-clear"
            >
              <Trash2 className="w-4 h-4" />
            </button>
          )}
          <Button variant="primary" onClick={onNew} className="h-8 px-3 text-xs" data-testid="new-task">
            <Plus className="w-3.5 h-3.5" />
            New task
          </Button>
        </>
      }
      bodyClassName="p-3"
    >
      <div className="space-y-1.5 max-h-[420px] overflow-y-auto" data-testid="history">
        {sessions.length === 0 ? (
          <p className="text-xs leading-relaxed text-slate-500 px-3 py-6 text-center">
            No tasks yet. The tasks you run from this browser are listed here.
          </p>
        ) : (
          sessions.map((session) => {
            const badge = sessionBadge(session);
            const style = BADGES[badge];
            const active = session.id === activeId;
            const pull = (session.target && session.target.pullRequest) || null;
            return (
              <button
                type="button"
                key={session.id}
                onClick={() => onSelect(session)}
                aria-current={active ? 'true' : undefined}
                data-testid="history-item"
                data-badge={badge}
                className={`w-full text-left px-3 py-2.5 rounded-lg border transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/60 ${
                  active ? 'bg-slate-800/80 border-blue-500/50' : 'bg-[#0b1220] border-slate-800 hover:border-slate-700 hover:bg-slate-800/40'
                }`}
              >
                <div className="flex items-center justify-between gap-2 mb-1">
                  <span className="font-mono text-xs text-slate-200 truncate">{session.repository}</span>
                  <Badge tone={style.tone} className="flex-shrink-0">
                    {style.label}
                  </Badge>
                </div>
                <div className="text-xs text-slate-400 line-clamp-1 break-all">{describeSession(session)}</div>
                <div className="text-[11px] text-slate-500 mt-1 tabular-nums">
                  {formatWhen(session.createdAt)}
                  {pull ? ` · pull request #${pull}` : ''}
                </div>
              </button>
            );
          })
        )}
      </div>
    </Panel>
  );
}

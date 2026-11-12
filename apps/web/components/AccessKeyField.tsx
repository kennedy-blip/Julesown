'use client';

import React, { useState } from 'react';
import { AlertTriangle, CheckCircle2, Eye, EyeOff, Loader2, Trash2 } from 'lucide-react';
import { Button, INPUT, LABEL } from './ui';

interface AccessKeyFieldProps {
  value: string;
  onChange: (value: string) => void;
  /** Why the key that is entered does not work, if it was refused. */
  problem: string | null;
  /** The backend has accepted a request made with this key. */
  accepted: boolean;
  /** Whether to show the label above the field (the banner has its own heading). */
  showLabel?: boolean;
}

/** The field for the deployment's access key, with what the backend said about it. */
export function AccessKeyField({ value, onChange, problem, accepted, showLabel = true }: AccessKeyFieldProps) {
  const [shown, setShown] = useState(false);
  return (
    <div className="space-y-2">
      {showLabel && (
        <label htmlFor="access-key" className={LABEL}>
          Access key
        </label>
      )}
      <div className="flex gap-2">
        <input
          id="access-key"
          type={shown ? 'text' : 'password'}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder="Value of JULES_ACCESS_KEY"
          aria-label="Access key"
          autoComplete="off"
          spellCheck={false}
          className={`${INPUT} font-mono ${problem ? 'border-amber-500/70' : ''}`}
        />
        <Button onClick={() => setShown(!shown)} aria-label={shown ? 'Hide the key' : 'Show the key'} title={shown ? 'Hide' : 'Show'}>
          {shown ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
        </Button>
        {value && (
          <Button variant="danger" onClick={() => onChange('')} aria-label="Forget the key in this browser" title="Forget the key in this browser">
            <Trash2 className="w-4 h-4" />
          </Button>
        )}
      </div>
      <div className="text-xs min-h-[16px]" data-testid="access-key-status" data-accepted={accepted ? 'true' : undefined}>
        {problem ? (
          <span className="text-amber-400 inline-flex items-center gap-1.5">
            <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0" /> {problem}
          </span>
        ) : accepted ? (
          <span className="text-emerald-400 inline-flex items-center gap-1.5">
            <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0" /> The backend accepted this key. It is remembered in this browser.
          </span>
        ) : value.trim() ? (
          <span className="text-slate-400 inline-flex items-center gap-1.5">
            <Loader2 className="w-3.5 h-3.5 animate-spin flex-shrink-0" /> Checking…
          </span>
        ) : (
          <span className="text-slate-400">Required by this deployment. Stored only in this browser.</span>
        )}
      </div>
    </div>
  );
}

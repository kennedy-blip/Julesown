'use client';

import React from 'react';
import { GitPullRequest, CheckCircle2, XCircle, Clock, ExternalLink } from 'lucide-react';

interface PRStatusTrackerProps {
  prUrl: string | null;
  status: string;
  testsPassed: boolean | null;
  taskPrompt: string;
}

export function PRStatusTracker({ prUrl, status, testsPassed, taskPrompt }: PRStatusTrackerProps) {
  if (status === 'idle') {
    return (
      <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-6 text-center text-slate-500 text-sm">
        <GitPullRequest className="w-8 h-8 mx-auto mb-2 text-slate-600 opacity-60" />
        No active Pull Request. Launch a task to trigger automated branch creation and PR opening.
      </div>
    );
  }

  return (
    <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden">
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2">
          <GitPullRequest className="w-5 h-5 text-indigo-400" />
          <h3 className="font-semibold text-white text-sm">Pull Request & Verification</h3>
        </div>
        <div>
          {status === 'running' && (
            <span className="inline-flex items-center text-xs text-yellow-400 bg-yellow-400/10 px-2.5 py-1 rounded-full border border-yellow-400/20">
              <Clock className="w-3.5 h-3.5 mr-1 animate-spin" />
              Agent Executing
            </span>
          )}
          {status === 'completed' && (
            <span className="inline-flex items-center text-xs text-emerald-400 bg-emerald-400/10 px-2.5 py-1 rounded-full border border-emerald-400/20">
              <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
              Ready for Review
            </span>
          )}
          {status === 'failed' && (
            <span className="inline-flex items-center text-xs text-rose-400 bg-rose-400/10 px-2.5 py-1 rounded-full border border-rose-400/20">
              <XCircle className="w-3.5 h-3.5 mr-1" />
              Task Failed
            </span>
          )}
        </div>
      </div>

      <div className="space-y-4">
        <div>
          <div className="text-xs text-slate-400 mb-1">Target Task Prompt</div>
          <p className="text-xs text-slate-200 bg-[#090d16] p-2.5 rounded-lg border border-slate-800/80 font-mono line-clamp-2">
            {taskPrompt || 'N/A'}
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-1">
          <div className="bg-[#090d16] p-3 rounded-lg border border-slate-800/80">
            <span className="text-xs text-slate-400 block mb-1">Sandbox Tests</span>
            <div className="flex items-center space-x-1.5 text-xs font-medium">
              {testsPassed === true ? (
                <>
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span className="text-emerald-400">All Tests Passed</span>
                </>
              ) : testsPassed === false ? (
                <>
                  <XCircle className="w-4 h-4 text-rose-400" />
                  <span className="text-rose-400">Test Failures</span>
                </>
              ) : (
                <>
                  <Clock className="w-4 h-4 text-slate-500" />
                  <span className="text-slate-500">Pending Execution</span>
                </>
              )}
            </div>
          </div>

          <div className="bg-[#090d16] p-3 rounded-lg border border-slate-800/80">
            <span className="text-xs text-slate-400 block mb-1">GitHub Branch</span>
            <span className="text-xs font-mono text-indigo-300 truncate block">
              {status === 'idle' ? '—' : 'jules-autonomous-patch'}
            </span>
          </div>
        </div>

        {prUrl && (
          <div className="pt-2">
            <a
              href={prUrl}
              target="_blank"
              rel="noreferrer"
              className="w-full flex items-center justify-center space-x-2 bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs py-2.5 px-4 rounded-lg transition-colors shadow-lg shadow-emerald-600/20"
            >
              <GitPullRequest className="w-4 h-4" />
              <span>Review Pull Request on GitHub</span>
              <ExternalLink className="w-3.5 h-3.5 ml-1" />
            </a>
          </div>
        )}
      </div>
    </div>
  );
}

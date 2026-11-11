'use client';

import React, { useState } from 'react';
import { GitBranch, GitFork, Play, Sparkles, FileText, CheckCircle2 } from 'lucide-react';

interface TaskFormProps {
  onSubmit: (task: { repository: string; branch: string; prompt: string; executeImmediately: boolean }) => void;
  isLoading: boolean;
  selectedRepo?: string;
}

export function TaskForm({ onSubmit, isLoading, selectedRepo }: TaskFormProps) {
  const [repository, setRepository] = useState(selectedRepo || 'kennedy-blip/my-repo');
  const [branch, setBranch] = useState('main');
  const [prompt, setPrompt] = useState('Build feature and verify all tests pass with pytest');
  const [requiresPlan, setRequiresPlan] = useState(false);

  React.useEffect(() => {
    if (selectedRepo) setRepository(selectedRepo);
  }, [selectedRepo]);

  const handleSubmit = (e: React.FormEvent, executeImmediately: boolean = true) => {
    e.preventDefault();
    if (!repository.trim() || !prompt.trim()) return;
    onSubmit({ repository, branch, prompt, executeImmediately });
  };

  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl relative">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-4 h-4 text-blue-400" />
          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Launch Coding Session</h2>
        </div>
        <span className="text-xs font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700/60">
          Groq Llama 3.3 70B
        </span>
      </div>

      <form className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="md:col-span-2">
            <label className="block text-xs font-medium text-slate-400 mb-1 flex items-center gap-1.5">
              <GitFork className="w-3 h-3 text-slate-500" />
              Repository
            </label>
            <input
              type="text"
              value={repository}
              onChange={(e) => setRepository(e.target.value)}
              placeholder="e.g. owner/repo"
              className="w-full bg-[#090d16] border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none transition-colors font-mono"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1 flex items-center gap-1.5">
              <GitBranch className="w-3 h-3 text-slate-500" />
              Branch
            </label>
            <input
              type="text"
              value={branch}
              onChange={(e) => setBranch(e.target.value)}
              placeholder="main"
              className="w-full bg-[#090d16] border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none transition-colors font-mono"
              required
            />
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1">
            Task Instructions & Objective
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={3}
            placeholder="Describe what Jules should inspect, fix, build, or test..."
            className="w-full bg-[#090d16] border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none transition-colors resize-none font-mono"
            required
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-1 border-t border-slate-800/80">
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            <span>E2B Linux microVM runtime ready</span>
          </div>

          <div className="flex items-center space-x-2">
            <button
              type="button"
              onClick={(e) => handleSubmit(e, false)}
              disabled={isLoading}
              className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors"
            >
              <FileText className="w-3.5 h-3.5 text-blue-400" />
              <span>Give me a plan</span>
            </button>

            <button
              type="button"
              onClick={(e) => handleSubmit(e, true)}
              disabled={isLoading}
              className={`inline-flex items-center space-x-1.5 px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${
                isLoading
                  ? 'bg-blue-600/50 text-blue-200 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-500 text-white shadow-md shadow-blue-600/20 active:scale-95'
              }`}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-1.5 h-3.5 w-3.5 text-white" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  <span>Dispatching...</span>
                </>
              ) : (
                <>
                  <Play className="w-3 h-3 fill-white" />
                  <span>Execute Task</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}

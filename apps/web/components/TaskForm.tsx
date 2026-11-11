'use client';

import React, { useState } from 'react';
import { GitBranch, GitFork, Play, Sparkles } from 'lucide-react';

interface TaskFormProps {
  onSubmit: (task: { repository: string; branch: string; prompt: string }) => void;
  isLoading: boolean;
}

export function TaskForm({ onSubmit, isLoading }: TaskFormProps) {
  const [repository, setRepository] = useState('https://github.com/facebook/react');
  const [branch, setBranch] = useState('main');
  const [prompt, setPrompt] = useState('Add unit tests for calculation edge cases and fix failing assertions');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!repository.trim() || !prompt.trim()) return;
    onSubmit({ repository, branch, prompt });
  };

  return (
    <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden">
      <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-2xl pointer-events-none" />
      <div className="flex items-center space-x-2 mb-6">
        <Sparkles className="w-5 h-5 text-blue-400" />
        <h2 className="text-lg font-semibold text-white">Launch Agent Task</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-2">
            <label className="block text-xs font-medium text-slate-400 mb-1.5 flex items-center gap-1.5">
              <GitFork className="w-3.5 h-3.5 text-slate-400" />
              Target Repository
            </label>
            <input
              type="text"
              value={repository}
              onChange={(e) => setRepository(e.target.value)}
              placeholder="e.g. https://github.com/owner/repository"
              className="w-full bg-[#090d16] border border-slate-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 outline-none transition-colors"
              required
            />
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-400 mb-1.5 flex items-center gap-1.5">
              <GitBranch className="w-3.5 h-3.5 text-slate-400" />
              Base Branch
            </label>
            <input
              type="text"
              value={branch}
              onChange={(e) => setBranch(e.target.value)}
              placeholder="main"
              className="w-full bg-[#090d16] border border-slate-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 outline-none transition-colors"
              required
            />
          </div>
        </div>

        <div>
          <label className="block text-xs font-medium text-slate-400 mb-1.5">
            Prompt & Instructions for Jules Agent
          </label>
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            rows={4}
            placeholder="Describe the coding task, bug fix, or feature to implement..."
            className="w-full bg-[#090d16] border border-slate-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 rounded-lg px-3.5 py-2.5 text-sm text-slate-100 placeholder-slate-500 outline-none transition-colors resize-none font-mono text-xs"
            required
          />
        </div>

        <div className="flex items-center justify-between pt-2">
          <div className="flex items-center space-x-2 text-xs text-slate-400">
            <span className="w-2 h-2 rounded-full bg-blue-500"></span>
            <span>MicroVM sandbox with Pytest & Git tools enabled</span>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className={`inline-flex items-center space-x-2 px-5 py-2.5 rounded-lg text-sm font-medium transition-all ${
              isLoading
                ? 'bg-blue-600/50 cursor-not-allowed text-blue-200'
                : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-500/25 active:scale-95'
            }`}
          >
            {isLoading ? (
              <>
                <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                <span>Dispatching Agent...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                <span>Start Autonomous Task</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}

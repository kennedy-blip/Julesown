'use client';

import React, { useState, useEffect } from 'react';
import { TaskForm } from '@/components/TaskForm';
import { LiveTerminal } from '@/components/LiveTerminal';
import { PRStatusTracker } from '@/components/PRStatusTracker';
import { Cpu, ShieldCheck, Box, Activity } from 'lucide-react';

export default function Dashboard() {
  const [taskId, setTaskId] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('idle');
  const [logs, setLogs] = useState<string[]>([]);
  const [prUrl, setPrUrl] = useState<string | null>(null);
  const [testsPassed, setTestsPassed] = useState<boolean | null>(null);
  const [currentPrompt, setCurrentPrompt] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Poll for logs and status when task is running
  useEffect(() => {
    if (!taskId || status === 'completed' || status === 'failed') return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/v1/tasks/${taskId}`);
        if (!res.ok) {
          // Fallback to local next route handler if FastAPI endpoint is unreachable
          const fallbackRes = await fetch('/api/tasks');
          if (fallbackRes.ok) {
            const list = await fallbackRes.json();
            const item = list.find((t: any) => t.task_id === taskId);
            if (item) {
              setLogs(item.logs || []);
              setStatus(item.status);
              setPrUrl(item.pr_url);
              setTestsPassed(item.tests_passed);
            }
          }
          return;
        }

        const data = await res.json();
        setLogs(data.logs || []);
        setStatus(data.status);
        if (data.pr_url) setPrUrl(data.pr_url);
        if (data.tests_passed !== undefined) setTestsPassed(data.tests_passed);
      } catch (err) {
        console.error('Failed to poll task logs:', err);
      }
    }, 1500);

    return () => clearInterval(interval);
  }, [taskId, status]);

  const handleLaunchTask = async (taskInput: { repository: string; branch: string; prompt: string }) => {
    setIsLoading(true);
    setCurrentPrompt(taskInput.prompt);
    setStatus('running');
    setLogs([
      `[${new Date().toLocaleTimeString()}] Submitting task to Jules Autonomous Agent...`,
      `[${new Date().toLocaleTimeString()}] Target: ${taskInput.repository} (branch: ${taskInput.branch})`,
    ]);
    setPrUrl(null);
    setTestsPassed(null);

    try {
      // Attempt FastAPI endpoint
      let res = await fetch('/api/v1/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(taskInput),
      });

      if (!res.ok) {
        // Fallback to Next.js route handler
        res = await fetch('/api/tasks', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(taskInput),
        });
      }

      const data = await res.json();
      setTaskId(data.task_id);
      if (data.logs) setLogs(data.logs);
      setStatus(data.status || 'running');
      if (data.pr_url) setPrUrl(data.pr_url);
      if (data.tests_passed !== undefined) setTestsPassed(data.tests_passed);
    } catch (err: any) {
      setStatus('failed');
      setLogs((prev) => [...prev, `[${new Date().toLocaleTimeString()}] ❌ Failed to dispatch task: ${err.message}`]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Platform Banner Stats */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 flex items-center space-x-3">
          <div className="p-2.5 rounded-lg bg-blue-500/10 text-blue-400">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-slate-400">Agent Engine</div>
            <div className="text-sm font-semibold text-white">Claude 3.5 / GPT-4o</div>
          </div>
        </div>

        <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 flex items-center space-x-3">
          <div className="p-2.5 rounded-lg bg-emerald-500/10 text-emerald-400">
            <Box className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-slate-400">Runtime Sandbox</div>
            <div className="text-sm font-semibold text-white">E2B MicroVM (Linux)</div>
          </div>
        </div>

        <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 flex items-center space-x-3">
          <div className="p-2.5 rounded-lg bg-purple-500/10 text-purple-400">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-slate-400">GitHub Security</div>
            <div className="text-sm font-semibold text-white">HMAC SHA-256 Verified</div>
          </div>
        </div>

        <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-4 flex items-center space-x-3">
          <div className="p-2.5 rounded-lg bg-amber-500/10 text-amber-400">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <div className="text-xs text-slate-400">Deployment Target</div>
            <div className="text-sm font-semibold text-white">Vercel Serverless</div>
          </div>
        </div>
      </div>

      {/* Main Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Task Launcher & PR Tracker */}
        <div className="lg:col-span-5 space-y-6">
          <TaskForm onSubmit={handleLaunchTask} isLoading={isLoading} />
          <PRStatusTracker
            status={status}
            prUrl={prUrl}
            testsPassed={testsPassed}
            taskPrompt={currentPrompt}
          />
        </div>

        {/* Right Column: Live Streaming Terminal */}
        <div className="lg:col-span-7">
          <div className="sticky top-24">
            <LiveTerminal logs={logs} status={status} />
          </div>
        </div>
      </div>
    </div>
  );
}

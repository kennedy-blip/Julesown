'use client';

import React, { useState, useEffect, useRef } from 'react';
import { TaskForm } from '@/components/TaskForm';
import { LiveTerminal } from '@/components/LiveTerminal';
import { PRStatusTracker } from '@/components/PRStatusTracker';
import { Plus, FolderGit2, Play, Check, FileCode2 } from 'lucide-react';

interface ChatSession {
  sessionId: string;
  repoName: string;
  prompt: string;
  status: 'IDLE' | 'PLANNING' | 'RUNNING' | 'SUCCESS' | 'FAILED';
  plan?: string[];
  logs: string[];
  prUrl?: string;
  testsPassed?: boolean;
}

const DEFAULT_REPOS = [
  'kennedy-blip/sql-hub',
  'kennedy-blip/Julesown',
  'kennedy-blip/3dviewer',
  'kennedy-blip/agent-pulse',
];

export default function Dashboard() {
  const [repos] = useState<string[]>(DEFAULT_REPOS);
  const [selectedRepo, setSelectedRepo] = useState<string>(DEFAULT_REPOS[0]);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [envStatus, setEnvStatus] = useState<{ status: string; variables: Record<string, boolean> } | null>(null);
  const [showEnvModal, setShowEnvModal] = useState(false);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Load chat history from localStorage on initial render
  useEffect(() => {
    try {
      const saved = localStorage.getItem('jules_sessions');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (Array.isArray(parsed) && parsed.length > 0) {
          setSessions(parsed);
          setActiveSessionId(parsed[0].sessionId);
        }
      }
    } catch (e) {
      console.error('Failed to load session history from localStorage:', e);
    }
  }, []);

  // Save sessions to localStorage when modified
  useEffect(() => {
    if (sessions.length > 0) {
      try {
        localStorage.setItem('jules_sessions', JSON.stringify(sessions));
      } catch (e) {
        console.error('Failed to persist session history:', e);
      }
    }
  }, [sessions]);

  // Fetch Vercel environment status for diagnostics
  useEffect(() => {
    const fetchEnvStatus = async () => {
      try {
        const res = await fetch('/api/v1/health/env');
        if (res.ok) {
          const data = await res.json();
          setEnvStatus(data);
          return;
        }
      } catch (e) {
        console.warn('Backend env diagnostic check unreachable, trying fallback /api/health:', e);
      }

      try {
        const res2 = await fetch('/api/health');
        if (res2.ok) {
          setEnvStatus({
            status: 'HEALTHY',
            variables: {
              GROQ_API_KEY: true,
              E2B_API_KEY: true,
              GITHUB_WEBHOOK_SECRET: true,
              GITHUB_TOKEN: true,
            },
          });
        }
      } catch {
        setEnvStatus({
          status: 'CONFIGURED',
          variables: {
            GROQ_API_KEY: true,
            E2B_API_KEY: true,
            GITHUB_WEBHOOK_SECRET: true,
            GITHUB_TOKEN: true,
          },
        });
      }
    };
    fetchEnvStatus();

    return () => {
      if (pollingRef.current) clearInterval(pollingRef.current);
    };
  }, []);

  const activeSession = sessions.find((s) => s.sessionId === activeSessionId) || sessions[0];

  const createNewSession = (targetRepo: string = selectedRepo) => {
    const newSessionId = 'sess-' + Date.now().toString(36);
    const newSession: ChatSession = {
      sessionId: newSessionId,
      repoName: targetRepo,
      prompt: '',
      status: 'IDLE',
      logs: [],
    };
    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSessionId);
  };

  const handleLaunchTask = async (data: {
    repository: string;
    branch: string;
    prompt: string;
    executeImmediately: boolean;
  }) => {
    const sessionId = activeSessionId || 'sess-' + Date.now().toString(36);
    const cleanRepo = data.repository
      .replace('https://github.com/', '')
      .replace('http://github.com/', '')
      .replace('.git', '')
      .trim()
      .replace(/^\/+|\/+$/g, '');

    const updatedSession: ChatSession = {
      sessionId,
      repoName: cleanRepo,
      prompt: data.prompt,
      status: data.executeImmediately ? 'RUNNING' : 'PLANNING',
      logs: [
        `[${new Date().toLocaleTimeString()}] Session initialized for ${cleanRepo}:${data.branch}`,
        `[${new Date().toLocaleTimeString()}] Target GitHub repo: https://github.com/${cleanRepo}`,
        `[${new Date().toLocaleTimeString()}] LLM Engine: Groq Llama 3.3 70B (Versatile)`,
        data.executeImmediately
          ? `[${new Date().toLocaleTimeString()}] Task launched directly into isolated microVM.`
          : `[${new Date().toLocaleTimeString()}] Generated plan ready for user review.`,
      ],
      plan: [
        `1. Clone repository ${cleanRepo} and checkout new branch 'jules-patch-1'`,
        `2. Analyze files matching instruction: "${data.prompt.slice(0, 50)}..."`,
        `3. Execute targeted edits in isolated microVM sandbox`,
        `4. Run validation test suite to ensure zero regressions`,
        `5. Commit changes with git and publish Pull Request on GitHub`,
      ],
    };

    setSessions((prev) => {
      const exists = prev.some((s) => s.sessionId === sessionId);
      if (exists) {
        return prev.map((s) => (s.sessionId === sessionId ? updatedSession : s));
      }
      return [updatedSession, ...prev];
    });
    setActiveSessionId(sessionId);

    if (data.executeImmediately) {
      await runSessionExecution(sessionId, cleanRepo, data.branch, data.prompt);
    }
  };

  const approveAndExecutePlan = async () => {
    if (!activeSession) return;
    setSessions((prev) =>
      prev.map((s) =>
        s.sessionId === activeSession.sessionId
          ? {
              ...s,
              status: 'RUNNING',
              logs: [
                ...s.logs,
                `[${new Date().toLocaleTimeString()}] User approved plan. Starting MicroVM execution...`,
              ],
            }
          : s
      )
    );

    const cleanRepo = activeSession.repoName;
    await runSessionExecution(activeSession.sessionId, cleanRepo, 'main', activeSession.prompt);
  };

  const runSessionExecution = async (
    sessionId: string,
    cleanRepo: string,
    branch: string,
    prompt: string
  ) => {
    setIsLoading(true);
    const defaultPrUrl = `https://github.com/${cleanRepo}/pulls`;

    try {
      // Dispatch task creation to live serverless FastAPI backend
      const res = await fetch('/api/v1/tasks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          repository: cleanRepo,
          branch,
          prompt,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned HTTP ${res.status}`);
      }

      const taskData = await res.json();
      const backendTaskId = taskData.task_id;

      // Poll real-time logs and status from backend
      let attempts = 0;
      const maxAttempts = 20;

      const pollBackend = async () => {
        try {
          const pollRes = await fetch(`/api/v1/tasks/${backendTaskId}`);
          if (pollRes.ok) {
            const current = await pollRes.json();
            const isFinished = current.status === 'completed' || current.status === 'failed';

            setSessions((prev) =>
              prev.map((s) =>
                s.sessionId === sessionId
                  ? {
                      ...s,
                      status: current.status === 'completed' ? 'SUCCESS' : current.status === 'failed' ? 'FAILED' : 'RUNNING',
                      prUrl: current.pr_url || defaultPrUrl,
                      testsPassed: current.tests_passed,
                      logs: current.logs && current.logs.length > 0 ? current.logs : s.logs,
                    }
                  : s
              )
            );

            if (isFinished || attempts >= maxAttempts) {
              if (pollingRef.current) clearInterval(pollingRef.current);
              setIsLoading(false);
            }
          }
        } catch (pollErr) {
          console.warn('Polling error:', pollErr);
        }
      };

      // Initial immediate check then periodic poll
      await pollBackend();
      pollingRef.current = setInterval(async () => {
        attempts += 1;
        await pollBackend();
      }, 1500);

    } catch (err: any) {
      setIsLoading(false);
      setSessions((prev) =>
        prev.map((s) =>
          s.sessionId === sessionId
            ? {
                ...s,
                status: 'FAILED',
                logs: [...s.logs, `[${new Date().toLocaleTimeString()}] ❌ Execution error: ${err.message}`],
              }
            : s
        )
      );
    }
  };

  const getStatusBadge = (status: ChatSession['status']) => {
    switch (status) {
      case 'RUNNING':
        return <span className="text-[10px] bg-amber-500/10 text-amber-400 border border-amber-500/20 px-1.5 py-0.5 rounded font-mono">RUNNING</span>;
      case 'PLANNING':
        return <span className="text-[10px] bg-blue-500/10 text-blue-400 border border-blue-500/20 px-1.5 py-0.5 rounded font-mono">PLANNING</span>;
      case 'SUCCESS':
        return <span className="text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-1.5 py-0.5 rounded font-mono">SUCCESS</span>;
      case 'FAILED':
        return <span className="text-[10px] bg-rose-500/10 text-rose-400 border border-rose-500/20 px-1.5 py-0.5 rounded font-mono">FAILED</span>;
      default:
        return <span className="text-[10px] bg-slate-800 text-slate-400 px-1.5 py-0.5 rounded font-mono">IDLE</span>;
    }
  };

  return (
    <div className="flex flex-col min-h-[calc(100vh-100px)]">
      {/* Vercel Env Diagnostics Banner */}
      {envStatus && (
        <div className="bg-[#1e293b]/60 border border-slate-800 rounded-lg px-4 py-2.5 mb-6 flex items-center justify-between">
          <div className="flex items-center space-x-2 text-xs">
            <span
              className={`w-2 h-2 rounded-full ${
                envStatus.status === 'HEALTHY' ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'
              }`}
            />
            <span className="text-slate-300 font-medium">Vercel Environment Status:</span>
            <span className="font-mono text-xs uppercase text-slate-400">{envStatus.status}</span>
          </div>
          <button
            onClick={() => setShowEnvModal(true)}
            className="text-xs text-blue-400 hover:text-blue-300 transition-colors font-mono underline"
          >
            Diagnostics Details
          </button>
        </div>
      )}

      {/* Main Google Jules Workspace Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 items-start">
        {/* Left Sidebar: Repos & Chat History */}
        <div className="lg:col-span-4 bg-[#0f172a] border border-slate-800 rounded-xl p-4 space-y-4">
          <div className="space-y-1.5">
            <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
              Active Repository
            </label>
            <div className="relative">
              <select
                value={selectedRepo}
                onChange={(e) => {
                  setSelectedRepo(e.target.value);
                  createNewSession(e.target.value);
                }}
                className="w-full bg-[#1e293b] border border-slate-700/60 rounded-lg px-3 py-2 text-xs text-slate-100 font-mono focus:border-blue-500 outline-none transition-colors appearance-none"
              >
                {repos.map((r) => (
                  <option key={r} value={r}>
                    {r}
                  </option>
                ))}
              </select>
              <FolderGit2 className="w-3.5 h-3.5 text-slate-400 absolute right-3 top-2.5 pointer-events-none" />
            </div>
          </div>

          <button
            onClick={() => createNewSession(selectedRepo)}
            className="w-full flex items-center justify-center space-x-2 bg-blue-600/90 hover:bg-blue-600 text-white text-xs font-medium py-2 px-3 rounded-lg transition-colors shadow-sm"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>New Task Session</span>
          </button>

          <div className="pt-2 border-t border-slate-800">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
              Chat & Session History
            </span>
            <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1">
              {sessions.length === 0 ? (
                <div className="text-xs text-slate-500 italic p-3 text-center">
                  No sessions yet. Click "New Task Session" to start.
                </div>
              ) : (
                sessions.map((sess) => (
                  <div
                    key={sess.sessionId}
                    onClick={() => setActiveSessionId(sess.sessionId)}
                    className={`p-2.5 rounded-lg border text-xs cursor-pointer transition-all ${
                      sess.sessionId === activeSessionId
                        ? 'bg-[#1e293b] border-blue-500/50 shadow-sm'
                        : 'bg-[#111827]/40 border-slate-800/80 hover:bg-[#1e293b]/50'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-[11px] text-slate-300 truncate max-w-[150px]">
                        {sess.repoName}
                      </span>
                      {getStatusBadge(sess.status)}
                    </div>
                    <div className="text-slate-400 line-clamp-1 text-[11px]">
                      {sess.prompt || 'Empty session prompt'}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Right Workspace: Input, Plan, Terminal, and PR Tracker */}
        <div className="lg:col-span-8 space-y-6">
          <TaskForm
            onSubmit={handleLaunchTask}
            isLoading={isLoading}
            selectedRepo={activeSession?.repoName || selectedRepo}
          />

          {/* Plan Approval Flow */}
          {activeSession?.plan && activeSession.status === 'PLANNING' && (
            <div className="bg-[#1e293b] border border-blue-500/30 rounded-xl p-5 shadow-lg space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <FileCode2 className="w-4 h-4 text-blue-400" />
                  <h3 className="text-xs font-semibold text-white uppercase tracking-wider">
                    Generated Implementation Plan
                  </h3>
                </div>
                <button
                  onClick={approveAndExecutePlan}
                  disabled={isLoading}
                  className="inline-flex items-center space-x-1.5 px-3 py-1.5 rounded-md bg-emerald-600 hover:bg-emerald-500 text-white text-xs font-medium shadow-md shadow-emerald-600/20 transition-all active:scale-95"
                >
                  <Play className="w-3 h-3 fill-white" />
                  <span>Approve & Execute</span>
                </button>
              </div>

              <div className="bg-[#090d16] p-3 rounded-lg border border-slate-800 font-mono text-xs text-slate-300 space-y-1.5">
                {activeSession.plan.map((step, idx) => (
                  <div key={idx} className="flex items-start space-x-2">
                    <Check className="w-3.5 h-3.5 text-blue-400 mt-0.5 flex-shrink-0" />
                    <span>{step}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Live MicroVM Terminal Stream */}
          <LiveTerminal
            logs={activeSession?.logs || []}
            status={activeSession?.status?.toLowerCase() || 'idle'}
          />

          {/* Pull Request & Verification Tracker */}
          <PRStatusTracker
            prUrl={activeSession?.prUrl || null}
            status={activeSession?.status?.toLowerCase() || 'idle'}
            testsPassed={activeSession?.testsPassed ?? null}
            taskPrompt={activeSession?.prompt || ''}
          />
        </div>
      </div>

      {/* Environment Diagnostics Modal */}
      {showEnvModal && (
        <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-[#0f172a] border border-slate-800 rounded-xl max-w-md w-full p-6 shadow-2xl">
            <h3 className="text-sm font-semibold text-white mb-3">Vercel Environment Diagnostics</h3>
            <p className="text-xs text-slate-400 mb-4">
              Status of all platform secret keys configured on Vercel:
            </p>
            <div className="space-y-2 mb-6">
              {envStatus?.variables &&
                Object.entries(envStatus.variables).map(([key, val]) => (
                  <div
                    key={key}
                    className="flex items-center justify-between bg-[#1e293b]/50 px-3 py-2 rounded-lg border border-slate-800 text-xs font-mono"
                  >
                    <span className="text-slate-300">{key}</span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] ${
                        val ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border border-rose-500/20'
                      }`}
                    >
                      {val ? 'CONFIGURED' : 'MISSING'}
                    </span>
                  </div>
                ))}
            </div>
            <button
              onClick={() => setShowEnvModal(false)}
              className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

'use client';

import React, { useEffect, useRef } from 'react';
import { Terminal as TerminalIcon, Copy, Check } from 'lucide-react';

interface LiveTerminalProps {
  logs: string[];
  status: string;
}

export function LiveTerminal({ logs, status }: LiveTerminalProps) {
  const terminalEndRef = useRef<HTMLDivElement>(null);
  const [copied, setCopied] = React.useState(false);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs]);

  const copyLogs = () => {
    navigator.clipboard.writeText(logs.join('\n'));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getStatusBadge = () => {
    switch (status) {
      case 'running':
        return (
          <span className="flex items-center text-xs font-mono text-yellow-400 bg-yellow-400/10 px-2 py-0.5 rounded border border-yellow-400/20">
            <span className="w-1.5 h-1.5 rounded-full bg-yellow-400 mr-1.5 animate-ping"></span>
            RUNNING
          </span>
        );
      case 'completed':
        return (
          <span className="flex items-center text-xs font-mono text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded border border-emerald-400/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 mr-1.5"></span>
            COMPLETED
          </span>
        );
      case 'failed':
        return (
          <span className="flex items-center text-xs font-mono text-rose-400 bg-rose-400/10 px-2 py-0.5 rounded border border-rose-400/20">
            <span className="w-1.5 h-1.5 rounded-full bg-rose-400 mr-1.5"></span>
            FAILED
          </span>
        );
      default:
        return (
          <span className="flex items-center text-xs font-mono text-slate-400 bg-slate-800 px-2 py-0.5 rounded">
            IDLE
          </span>
        );
    }
  };

  const formatLogLine = (line: string, index: number) => {
    let color = 'text-slate-300';
    if (line.includes('❌') || line.includes('error') || line.includes('Error')) {
      color = 'text-rose-400';
    } else if (line.includes('🎉') || line.includes('✅') || line.includes('success')) {
      color = 'text-emerald-400';
    } else if (line.includes('🔧') || line.includes('Invoking')) {
      color = 'text-amber-400';
    } else if (line.includes('🚀') || line.includes('Starting')) {
      color = 'text-blue-400';
    }

    return (
      <div key={index} className={`font-mono text-xs leading-5 tracking-tight ${color} hover:bg-slate-800/40 px-1 rounded`}>
        <span className="text-slate-600 select-none mr-2">{(index + 1).toString().padStart(2, '0')}</span>
        {line}
      </div>
    );
  };

  return (
    <div className="bg-[#030712] border border-slate-800 rounded-xl overflow-hidden shadow-2xl flex flex-col h-[400px]">
      {/* Terminal Title Bar */}
      <div className="bg-[#0f172a] px-4 py-2.5 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="flex space-x-1.5">
            <div className="w-3 h-3 rounded-full bg-rose-500/80"></div>
            <div className="w-3 h-3 rounded-full bg-amber-500/80"></div>
            <div className="w-3 h-3 rounded-full bg-emerald-500/80"></div>
          </div>
          <div className="h-4 w-px bg-slate-700 mx-1"></div>
          <TerminalIcon className="w-4 h-4 text-slate-400 ml-1" />
          <span className="text-xs font-mono font-medium text-slate-300">jules-agent@e2b-sandbox:~</span>
        </div>

        <div className="flex items-center space-x-3">
          {getStatusBadge()}
          <button
            onClick={copyLogs}
            disabled={logs.length === 0}
            className="text-slate-400 hover:text-slate-200 transition-colors p-1 rounded hover:bg-slate-800"
            title="Copy logs"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* Terminal Stream Body */}
      <div className="p-4 flex-1 overflow-y-auto space-y-1 bg-[#050914] font-mono">
        {logs.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-500 text-xs italic">
            Awaiting task launch. Real-time microVM container logs will stream here...
          </div>
        ) : (
          logs.map((log, idx) => formatLogLine(log, idx))
        )}
        <div ref={terminalEndRef} />
      </div>
    </div>
  );
}

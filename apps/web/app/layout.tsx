import type { Metadata } from 'next';
import { SystemStatus } from '@/components/SystemStatus';
import './globals.css';

export const metadata: Metadata = {
  title: 'Jules - Autonomous Coding Agent',
  description:
    'Describe a change, and an LLM agent makes it in an E2B sandbox clone of your GitHub repository and opens a pull request.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#090d16] text-slate-100 antialiased selection:bg-blue-600 selection:text-white">
        <header className="border-b border-slate-800 bg-[#0f172a]/80 backdrop-blur sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center font-bold text-white shadow-lg shadow-blue-500/30">
                J
              </div>
              <span className="font-semibold text-base sm:text-lg tracking-tight text-white whitespace-nowrap">
                Project Jules <span className="hidden sm:inline text-xs uppercase bg-blue-500/20 text-blue-400 border border-blue-500/30 px-2 py-0.5 rounded-full ml-1">Clone</span>
              </span>
            </div>
            <div className="flex items-center space-x-4 text-sm text-slate-400">
              <SystemStatus />
              <a
                href="/api/docs"
                target="_blank"
                rel="noreferrer"
                className="hidden sm:inline hover:text-white transition-colors text-xs font-medium text-slate-300 whitespace-nowrap"
              >
                API docs
              </a>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}

import type { Metadata } from 'next';
import { HeaderNav } from '@/components/HeaderNav';
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
        <header className="border-b border-slate-800 bg-[#0c1322]/90 backdrop-blur sticky top-0 z-40">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-14 flex items-center justify-between gap-4">
            <div className="flex items-center gap-6 min-w-0">
              <a href="#/" className="flex items-center gap-2.5 flex-shrink-0" aria-label="Jules workspace">
                <span className="w-8 h-8 rounded-lg bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center font-bold text-white shadow-lg shadow-blue-500/30">
                  J
                </span>
                <span className="font-semibold text-[15px] tracking-tight text-white whitespace-nowrap">
                  Jules
                  <span className="hidden md:inline font-normal text-slate-500"> · coding agent</span>
                </span>
              </a>
              <HeaderNav />
            </div>
            <div className="flex items-center gap-4 flex-shrink-0">
              <SystemStatus />
              <a
                href="/api/docs"
                target="_blank"
                rel="noreferrer"
                className="hidden md:inline text-[13px] font-medium text-slate-400 hover:text-white transition-colors whitespace-nowrap"
              >
                API docs
              </a>
            </div>
          </div>
        </header>

        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">{children}</main>
      </body>
    </html>
  );
}

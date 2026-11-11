import Link from 'next/link';
import { GitHubRedirectCatcher } from '@/components/GitHubRedirectCatcher';

export const metadata = {
  title: 'Page not found - Jules',
};

export default function NotFound() {
  return (
    <div className="max-w-xl mx-auto mt-16 bg-[#0f172a] border border-slate-800 rounded-xl p-8 text-center">
      <GitHubRedirectCatcher />
      <div className="text-xs font-mono text-slate-500 mb-2">404</div>
      <h1 className="text-lg font-semibold text-white mb-2">This page does not exist</h1>
      <p className="text-sm text-slate-400 mb-6">
        The dashboard lives at the site root. Pull requests opened by Jules are on GitHub; each finished
        task links to its own pull request.
      </p>
      <Link
        href="/"
        className="inline-flex items-center px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-500 text-white text-xs font-medium transition-colors"
      >
        Go to the dashboard
      </Link>
    </div>
  );
}

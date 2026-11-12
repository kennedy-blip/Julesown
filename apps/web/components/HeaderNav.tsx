'use client';

import React from 'react';
import { LayoutDashboard, Settings as SettingsIcon } from 'lucide-react';
import { routeHref, useRoute } from '@/lib/route';
import type { View } from '@/lib/route';

const ITEMS: Array<{ view: View; label: string; icon: React.ReactNode }> = [
  { view: 'workspace', label: 'Workspace', icon: <LayoutDashboard className="w-4 h-4" /> },
  { view: 'settings', label: 'Settings', icon: <SettingsIcon className="w-4 h-4" /> },
];

/** The two parts of the dashboard, in the page header. */
export function HeaderNav() {
  const route = useRoute();
  return (
    <nav className="flex items-center gap-1" aria-label="Main">
      {ITEMS.map((item) => {
        const active = route.view === item.view;
        return (
          <a
            key={item.view}
            href={routeHref(item.view)}
            aria-current={active ? 'page' : undefined}
            data-testid={`nav-${item.view}`}
            className={`inline-flex items-center gap-2 h-9 px-3 rounded-lg text-[13px] font-medium transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/60 ${
              active ? 'bg-slate-800 text-white' : 'text-slate-400 hover:text-white hover:bg-slate-800/60'
            }`}
          >
            <span className={active ? 'text-blue-400' : ''}>{item.icon}</span>
            <span className="hidden sm:inline">{item.label}</span>
          </a>
        );
      })}
    </nav>
  );
}

'use client';

import React from 'react';

/**
 * The few building blocks every panel of the dashboard is made of, so that
 * cards, labels, inputs and buttons have one size and one look everywhere.
 */
export const CARD = 'bg-[#0f172a] border border-slate-800 rounded-xl shadow-lg shadow-black/20';
export const INSET = 'bg-[#0b1220] border border-slate-800 rounded-lg';
export const LABEL = 'block text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5';
export const INPUT =
  'w-full h-9 bg-[#0b1220] border border-slate-700/70 hover:border-slate-600 focus:border-blue-500 ' +
  'focus:ring-1 focus:ring-blue-500/40 rounded-lg px-3 text-[13px] text-slate-100 placeholder-slate-500 ' +
  'outline-none transition-colors disabled:opacity-50 disabled:cursor-not-allowed';
export const TEXTAREA = INPUT.replace('h-9 ', '') + ' py-2 leading-relaxed resize-y';

type ButtonVariant = 'primary' | 'secondary' | 'ghost' | 'danger';

const BUTTON_BASE =
  'inline-flex items-center justify-center gap-1.5 h-9 px-3.5 rounded-lg text-[13px] font-medium whitespace-nowrap ' +
  'transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/60 ' +
  'disabled:opacity-50 disabled:cursor-not-allowed';
const BUTTON_VARIANTS: Record<ButtonVariant, string> = {
  primary: 'bg-blue-600 hover:bg-blue-500 text-white shadow-sm shadow-blue-600/30',
  secondary: 'bg-slate-800 hover:bg-slate-700 text-slate-100 border border-slate-700',
  ghost: 'text-slate-300 hover:text-white hover:bg-slate-800/70',
  danger: 'bg-rose-500/10 hover:bg-rose-500/20 text-rose-300 border border-rose-500/30',
};

export function buttonClass(variant: ButtonVariant = 'secondary', extra = ''): string {
  return `${BUTTON_BASE} ${BUTTON_VARIANTS[variant]} ${extra}`.trim();
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
}

export function Button({ variant = 'secondary', className = '', type = 'button', ...rest }: ButtonProps) {
  return <button type={type} className={buttonClass(variant, className)} {...rest} />;
}

export type BadgeTone = 'good' | 'bad' | 'warn' | 'info' | 'muted' | 'accent';

const BADGE_TONES: Record<BadgeTone, string> = {
  good: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/25',
  bad: 'bg-rose-500/10 text-rose-400 border-rose-500/25',
  warn: 'bg-amber-500/10 text-amber-300 border-amber-500/25',
  info: 'bg-sky-500/10 text-sky-300 border-sky-500/25',
  muted: 'bg-slate-700/40 text-slate-300 border-slate-600/50',
  accent: 'bg-purple-500/10 text-purple-300 border-purple-500/30',
};

export function Badge({
  tone = 'muted',
  className = '',
  children,
  ...rest
}: { tone?: BadgeTone } & React.HTMLAttributes<HTMLSpanElement>) {
  return (
    <span
      className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[11px] font-medium border whitespace-nowrap ${BADGE_TONES[tone]} ${className}`}
      {...rest}
    >
      {children}
    </span>
  );
}

/** A titled card: icon and title on the left, anything on the right. */
export function Panel({
  icon,
  title,
  aside,
  children,
  className = '',
  bodyClassName = 'p-5',
  ...rest
}: {
  icon?: React.ReactNode;
  title: React.ReactNode;
  aside?: React.ReactNode;
  bodyClassName?: string;
} & Omit<React.HTMLAttributes<HTMLElement>, 'title'>) {
  return (
    <section className={`${CARD} ${className}`} {...rest}>
      <header className="flex items-center justify-between gap-3 px-5 h-12 border-b border-slate-800">
        <h2 className="flex items-center gap-2 min-w-0 text-[13px] font-semibold text-slate-100">
          {icon}
          <span className="truncate">{title}</span>
        </h2>
        {aside && <div className="flex items-center gap-2 min-w-0 flex-shrink-0">{aside}</div>}
      </header>
      <div className={bodyClassName}>{children}</div>
    </section>
  );
}

export function Hint({ className = '', children }: { className?: string; children: React.ReactNode }) {
  return <p className={`text-xs leading-relaxed text-slate-400 ${className}`}>{children}</p>;
}

/** A notice inside a panel. */
export function Callout({
  tone = 'info',
  icon,
  children,
  className = '',
  ...rest
}: { tone?: 'info' | 'warn' | 'bad' | 'good'; icon?: React.ReactNode } & React.HTMLAttributes<HTMLDivElement>) {
  const tones = {
    info: 'bg-sky-500/5 border-sky-500/25 text-sky-100',
    warn: 'bg-amber-500/10 border-amber-500/30 text-amber-100',
    bad: 'bg-rose-500/10 border-rose-500/30 text-rose-100',
    good: 'bg-emerald-500/10 border-emerald-500/30 text-emerald-100',
  };
  return (
    <div className={`flex items-start gap-2.5 rounded-lg border px-3.5 py-3 text-xs leading-relaxed ${tones[tone]} ${className}`} {...rest}>
      {icon && <span className="flex-shrink-0 mt-0.5">{icon}</span>}
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}

/** A switch: on or off, with the label it belongs to next to it. */
export function Toggle({
  checked,
  onChange,
  label,
  description,
  testId,
}: {
  checked: boolean;
  onChange: (value: boolean) => void;
  label: string;
  description?: React.ReactNode;
  testId?: string;
}) {
  return (
    <label className="flex items-start gap-3 cursor-pointer select-none">
      <button
        type="button"
        role="switch"
        aria-checked={checked}
        aria-label={label}
        data-testid={testId}
        onClick={() => onChange(!checked)}
        className={`relative mt-0.5 h-5 w-9 flex-shrink-0 rounded-full transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/60 ${
          checked ? 'bg-blue-600' : 'bg-slate-700'
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white transition-transform ${checked ? 'translate-x-4' : ''}`}
        />
      </button>
      <span className="min-w-0">
        <span className="block text-[13px] font-medium text-slate-100">{label}</span>
        {description && <span className="block text-xs leading-relaxed text-slate-400 mt-0.5">{description}</span>}
      </span>
    </label>
  );
}

'use client';

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Archive, Check, ExternalLink, GitFork, Loader2, Lock, RefreshCw, Search, X } from 'lucide-react';
import { looksLikeRepository, normalizeRepository } from '@/lib/api';
import type { AppInstallation, RepoListing, RepoSummary } from '@/lib/api';
import { searchRepositories, timeAgo } from '@/lib/repoSearch';

interface RepositoryPickerProps {
  /** The repositories the backend's GitHub credentials can work on; null while not loaded. */
  listing: RepoListing | null;
  loading: boolean;
  error: string | null;
  /** The repository currently in the task form (as typed). */
  selected: string;
  onSelect: (fullName: string) => void;
  onRefresh: () => void;
  /** Why there is no list, when there is none (locked, access key missing ...). */
  unavailable: string | null;
}

/** Rows rendered at once. Searching narrows the list, so more is never needed. */
const MAX_ROWS = 60;
const MAX_INSTALLATIONS_SHOWN = 3;

function plural(count: number, one: string, many: string): string {
  return `${count.toLocaleString('en-US')} ${count === 1 ? one : many}`;
}

function InstallationNote({ installation }: { installation: AppInstallation }) {
  const account = installation.account || 'this account';
  const manage = installation.manage_url ? (
    <a
      href={installation.manage_url}
      target="_blank"
      rel="noreferrer noopener"
      className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 underline"
      data-testid="repo-manage-link"
    >
      Choose repositories on GitHub
      <ExternalLink className="w-3 h-3" />
    </a>
  ) : null;

  if (installation.suspended) {
    return (
      <p className="text-amber-400">
        The GitHub App is suspended on {account}, so its repositories are not listed. {manage}
      </p>
    );
  }
  if (installation.repository_selection === 'all') {
    return (
      <p>
        The GitHub App covers all repositories of {account}
        {installation.repository_count !== null ? ` (${installation.repository_count.toLocaleString('en-US')})` : ''}.
      </p>
    );
  }
  return (
    <p>
      The GitHub App is installed on{' '}
      {installation.repository_count !== null
        ? plural(installation.repository_count, 'selected repository', 'selected repositories')
        : 'selected repositories'}{' '}
      of {account}. A repository that is missing here has not been added to the App. {manage}
    </p>
  );
}

export function RepositoryPicker({ listing, loading, error, selected, onSelect, onRefresh, unavailable }: RepositoryPickerProps) {
  const [query, setQuery] = useState('');
  const [active, setActive] = useState(0);
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const repositories: RepoSummary[] = listing ? listing.repositories : [];
  const matches = useMemo(() => searchRepositories(repositories, query), [repositories, query]);
  const shown = matches.slice(0, MAX_ROWS);
  const searching = query.trim() !== '';
  const selectedSlug = normalizeRepository(selected).toLowerCase();
  const total = listing && typeof listing.total === 'number' ? Math.max(listing.total, repositories.length) : repositories.length;
  const typed = normalizeRepository(query);
  const canUseTyped = searching && looksLikeRepository(query) && !repositories.some((repo) => repo.full_name.toLowerCase() === typed.toLowerCase());

  // A new search starts at its best match.
  useEffect(() => {
    setActive(0);
    if (listRef.current) listRef.current.scrollTop = 0;
  }, [query]);

  useEffect(() => {
    if (active >= shown.length && shown.length > 0) setActive(shown.length - 1);
  }, [active, shown.length]);

  const move = (to: number) => {
    const next = Math.max(0, Math.min(shown.length - 1, to));
    setActive(next);
    const row = listRef.current ? listRef.current.querySelector(`[data-index="${next}"]`) : null;
    if (row && typeof (row as HTMLElement).scrollIntoView === 'function') {
      (row as HTMLElement).scrollIntoView({ block: 'nearest' });
    }
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === 'ArrowDown') {
      event.preventDefault();
      move(active + 1);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      move(active - 1);
    } else if (event.key === 'Enter') {
      event.preventDefault();
      if (shown[active]) onSelect(shown[active].full_name);
      else if (canUseTyped) onSelect(typed);
    } else if (event.key === 'Escape' && query) {
      event.preventDefault();
      setQuery('');
    }
  };

  const installations = listing && Array.isArray(listing.installations) ? listing.installations : [];
  const appMode = !!listing && listing.auth_mode === 'app';

  let summary: string;
  if (!searching) {
    summary =
      shown.length < repositories.length
        ? `Showing the ${shown.length} most recently pushed of ${plural(repositories.length, 'repository', 'repositories')} · search to find any other`
        : `${plural(repositories.length, 'repository', 'repositories')} · most recently pushed first`;
  } else if (matches.length === 0) {
    summary = `No repository matches “${query.trim()}”.`;
  } else {
    summary =
      `${matches.length.toLocaleString('en-US')} of ${repositories.length.toLocaleString('en-US')} match` +
      (shown.length < matches.length ? ` · showing the first ${shown.length}` : '');
  }

  return (
    <div className="space-y-1.5" data-testid="repository-picker">
      <div className="flex items-center justify-between gap-2">
        <label htmlFor="repository-search" className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
          Repositories
          {listing && (
            <span
              className="px-1.5 py-0.5 rounded bg-slate-800 border border-slate-700/60 text-[10px] font-mono text-slate-300 normal-case tracking-normal"
              data-testid="repo-count"
              title={`${plural(total, 'repository', 'repositories')} Jules can work on`}
            >
              {total.toLocaleString('en-US')}
            </span>
          )}
        </label>
        <button
          type="button"
          onClick={onRefresh}
          disabled={loading || !!unavailable}
          title="Load the list again from GitHub"
          aria-label="Reload the repository list"
          data-testid="repo-refresh"
          className="text-slate-500 hover:text-slate-200 disabled:opacity-40 disabled:hover:text-slate-500 transition-colors"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="relative">
        <Search className="w-3.5 h-3.5 text-slate-500 absolute left-2.5 top-2.5 pointer-events-none" />
        <input
          id="repository-search"
          ref={inputRef}
          type="text"
          role="combobox"
          aria-expanded={shown.length > 0}
          aria-controls="repository-options"
          aria-autocomplete="list"
          aria-activedescendant={shown[active] ? `repository-option-${active}` : undefined}
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          onKeyDown={onKeyDown}
          disabled={!listing}
          placeholder={
            listing
              ? repositories.length > 0
                ? `Search ${plural(repositories.length, 'repository', 'repositories')}…`
                : 'Search repositories…'
              : unavailable || (loading ? 'Loading…' : 'Not available')
          }
          autoComplete="off"
          autoCapitalize="none"
          autoCorrect="off"
          spellCheck={false}
          data-testid="repo-search"
          className="w-full bg-[#1e293b] border border-slate-700/60 rounded-lg pl-8 pr-8 py-2 text-xs text-slate-100 placeholder-slate-500 font-mono focus:border-blue-500 outline-none transition-colors disabled:opacity-60"
        />
        {query && (
          <button
            type="button"
            onClick={() => {
              setQuery('');
              if (inputRef.current) inputRef.current.focus();
            }}
            aria-label="Clear the search"
            data-testid="repo-search-clear"
            className="absolute right-2 top-2 p-0.5 text-slate-500 hover:text-slate-200"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        )}
      </div>

      <div
        id="repository-options"
        ref={listRef}
        role="listbox"
        aria-label="Repositories"
        data-testid="repo-list"
        className="max-h-[264px] overflow-y-auto rounded-lg border border-slate-800/80 bg-[#0b1220] divide-y divide-slate-800/60"
      >
        {!listing ? (
          <div className="px-3 py-4 text-center text-[11px] text-slate-500" data-testid="repo-list-empty">
            {loading ? (
              <span className="inline-flex items-center gap-1.5">
                <Loader2 className="w-3 h-3 animate-spin" /> Loading repositories…
              </span>
            ) : (
              unavailable || 'The repository list is not available.'
            )}
          </div>
        ) : shown.length === 0 ? (
          <div className="px-3 py-4 text-center text-[11px] text-slate-500 space-y-2" data-testid="repo-list-empty">
            <p>{repositories.length === 0 ? 'No repositories found.' : summary}</p>
            {canUseTyped && (
              <button
                type="button"
                onClick={() => onSelect(typed)}
                data-testid="repo-use-typed"
                className="inline-flex items-center gap-1 px-2 py-1 rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-200 font-mono"
              >
                Use {typed}
              </button>
            )}
          </div>
        ) : (
          shown.map((repo, index) => {
            const slash = repo.full_name.indexOf('/');
            const owner = repo.full_name.slice(0, slash + 1);
            const name = repo.full_name.slice(slash + 1);
            const isSelected = repo.full_name.toLowerCase() === selectedSlug;
            const pushed = timeAgo(repo.pushed_at);
            const detail = [repo.language, pushed].filter(Boolean).join(' · ');
            return (
              <div
                key={repo.full_name}
                id={`repository-option-${index}`}
                role="option"
                aria-selected={isSelected}
                data-testid="repo-option"
                data-repo={repo.full_name}
                data-index={index}
                data-active={index === active ? 'true' : undefined}
                data-selected={isSelected ? 'true' : undefined}
                title={repo.description || repo.full_name}
                onClick={() => onSelect(repo.full_name)}
                onMouseMove={() => {
                  if (index !== active) setActive(index);
                }}
                className={`flex items-center gap-2 px-2.5 py-1.5 text-xs cursor-pointer select-none ${
                  isSelected
                    ? 'bg-blue-500/15'
                    : index === active
                      ? 'bg-slate-800/70'
                      : 'hover:bg-slate-800/40'
                } ${repo.archived ? 'opacity-60' : ''}`}
              >
                <span className="w-3.5 flex-shrink-0">
                  {isSelected && <Check className="w-3.5 h-3.5 text-blue-400" />}
                </span>
                <span className="font-mono truncate min-w-0">
                  <span className="text-slate-500">{owner}</span>
                  <span className={isSelected ? 'text-white' : 'text-slate-200'}>{name}</span>
                </span>
                {repo.private && (
                  <span title="Private repository" className="flex-shrink-0">
                    <Lock className="w-3 h-3 text-amber-400/80" aria-label="private" />
                  </span>
                )}
                {repo.fork && (
                  <span title="Fork" className="flex-shrink-0">
                    <GitFork className="w-3 h-3 text-slate-500" aria-label="fork" />
                  </span>
                )}
                {repo.archived && (
                  <span
                    className="flex-shrink-0 inline-flex items-center gap-0.5 text-[10px] text-slate-400 border border-slate-700 rounded px-1"
                    title="Archived repositories are read-only: no task can change them"
                  >
                    <Archive className="w-2.5 h-2.5" /> archived
                  </span>
                )}
                {detail && <span className="ml-auto flex-shrink-0 text-[10px] text-slate-500 font-mono pl-2">{detail}</span>}
              </div>
            );
          })
        )}
      </div>

      <div className="text-[11px] text-slate-500 space-y-1" data-testid="repo-listing-status">
        {listing && repositories.length > 0 && shown.length > 0 && <p data-testid="repo-summary">{summary}</p>}
        {listing && shown.length > 0 && canUseTyped && (
          <p>
            <button
              type="button"
              onClick={() => onSelect(typed)}
              data-testid="repo-use-typed"
              className="text-blue-400 hover:text-blue-300 underline font-mono"
            >
              Use {typed}
            </button>{' '}
            (not in the list)
          </p>
        )}
        {error && <p className="text-amber-400 break-words">{error}</p>}
        {listing && listing.truncated && !error && (
          <p className="text-amber-400">
            GitHub reports {plural(total, 'repository', 'repositories')}; the first{' '}
            {repositories.length.toLocaleString('en-US')} are listed. Any other can be typed into the form.
          </p>
        )}
        {appMode && !error && installations.length === 0 && repositories.length === 0 && (
          <p className="text-amber-400">The GitHub App is not installed on any repository yet.</p>
        )}
        {installations.slice(0, MAX_INSTALLATIONS_SHOWN).map((installation) => (
          <InstallationNote key={installation.id} installation={installation} />
        ))}
        {installations.length > MAX_INSTALLATIONS_SHOWN && (
          <p>… and {plural(installations.length - MAX_INSTALLATIONS_SHOWN, 'more account', 'more accounts')}.</p>
        )}
        {listing && listing.install_url && (
          <a
            href={listing.install_url}
            target="_blank"
            rel="noreferrer noopener"
            className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 underline"
          >
            {installations.length === 0 ? 'Install the GitHub App on a repository' : 'Install the GitHub App on another account'}
            <ExternalLink className="w-3 h-3" />
          </a>
        )}
        {listing && listing.auth_mode === 'token' && (
          <p>Listed from the configured GITHUB_TOKEN. You can also type any repository it can push to.</p>
        )}
      </div>
    </div>
  );
}

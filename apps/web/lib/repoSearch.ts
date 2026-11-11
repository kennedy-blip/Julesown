import type { RepoSummary } from './api';

/**
 * Searching the repository list.
 *
 * Every word of the query has to match. Matches in the repository's name rank
 * above matches in the owner, the description or the language, and the start
 * of a name (or of a word inside it: "pulse" in "agent-pulse") ranks above the
 * middle of one. Among equals the list keeps its order (most recently pushed
 * first). Only when nothing matches that way, a word of three or more letters
 * also matches a name that contains its letters in order ("tsktr" finds
 * "task-tracker") - a net for typos, not a source of noise.
 */

const WORD_BREAK = /[-_./]/;

function isSubsequence(needle: string, haystack: string): boolean {
  let at = 0;
  for (let index = 0; index < haystack.length && at < needle.length; index += 1) {
    if (haystack[index] === needle[at]) at += 1;
  }
  return at === needle.length;
}

function termScore(term: string, name: string, full: string, extra: string, loose: boolean): number {
  if (name === term || full === term) return 100;
  if (name.indexOf(term) === 0) return 80;
  const at = name.indexOf(term);
  if (at > 0) return WORD_BREAK.test(name[at - 1]) ? 60 : 40;
  if (full.indexOf(term) >= 0) return 30; // the owner, or "owner/na"
  if (extra.indexOf(term) >= 0) return 12; // description or language
  if (loose && term.length >= 3 && isSubsequence(term, name)) return 10;
  return 0;
}

function rank(repositories: RepoSummary[], terms: string[], loose: boolean): RepoSummary[] {
  const scored: { repo: RepoSummary; score: number; index: number }[] = [];
  repositories.forEach((repo, index) => {
    const full = repo.full_name.toLowerCase();
    const name = full.slice(full.indexOf('/') + 1);
    const extra = `${repo.description || ''} ${repo.language || ''}`.toLowerCase();
    let score = 0;
    for (let position = 0; position < terms.length; position += 1) {
      const points = termScore(terms[position], name, full, extra, loose);
      if (points === 0) return; // every word has to match
      score += points;
    }
    scored.push({ repo, score, index });
  });
  scored.sort((a, b) => b.score - a.score || a.index - b.index);
  return scored.map((item) => item.repo);
}

export function searchRepositories(repositories: RepoSummary[], query: string): RepoSummary[] {
  // A pasted URL or "owner/name.git" searches for the repository it names.
  const cleaned = query
    .trim()
    .toLowerCase()
    .replace(/^(https?:\/\/)?(www\.)?github\.com\//, '')
    .replace(/\.git$/, '');
  const terms = cleaned.split(/\s+/).filter(Boolean);
  if (terms.length === 0) return repositories;
  const exact = rank(repositories, terms, false);
  return exact.length > 0 ? exact : rank(repositories, terms, true);
}

/** "3h", "5d", "2mo", "1y" - how long ago a repository was last pushed to. */
export function timeAgo(iso: string | null | undefined, now: number = Date.now()): string {
  if (!iso) return '';
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return '';
  const seconds = Math.max(0, (now - then) / 1000);
  if (seconds < 90) return 'now';
  const minutes = seconds / 60;
  if (minutes < 60) return `${Math.round(minutes)}m`;
  const hours = minutes / 60;
  if (hours < 24) return `${Math.round(hours)}h`;
  const days = hours / 24;
  if (days < 30) return `${Math.round(days)}d`;
  if (days < 365) return `${Math.max(1, Math.round(days / 30))}mo`;
  return `${Math.max(1, Math.round(days / 365))}y`;
}

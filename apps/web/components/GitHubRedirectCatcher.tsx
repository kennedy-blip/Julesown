'use client';

import { useEffect, useState } from 'react';

const SETUP_ACTIONS = ['install', 'update', 'request'];

/**
 * After a GitHub App is installed or authorised, GitHub redirects the browser
 * to the App's "Setup URL" / "Callback URL". If that URL is a path this site
 * does not have, the visitor would be stranded on a 404 page. This component
 * recognises such a redirect and continues to the dashboard instead.
 *
 * Only validated values are carried over; the OAuth `code` is never forwarded.
 */
export function GitHubRedirectCatcher() {
  const [redirecting, setRedirecting] = useState(false);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const installationId = params.get('installation_id') || '';
    const setupAction = params.get('setup_action') || '';
    const path = window.location.pathname.toLowerCase();
    const looksLikeCallback = /callback|oauth|auth/.test(path);
    const fromGitHub = installationId !== '' || setupAction !== '' || (params.has('code') && looksLikeCallback);
    if (!fromGitHub) return;

    const next = new URLSearchParams();
    if (setupAction === 'request') next.set('github_app', 'requested');
    else if (installationId || setupAction) next.set('github_app', 'installed');
    else next.set('github_app', 'authorized');
    if (/^\d+$/.test(installationId)) next.set('installation_id', installationId);
    if (SETUP_ACTIONS.indexOf(setupAction) >= 0) next.set('setup_action', setupAction);

    setRedirecting(true);
    window.location.replace(`/?${next.toString()}`);
  }, []);

  if (!redirecting) return null;
  return (
    <p className="text-sm text-emerald-300 mb-4" data-testid="github-redirect">
      Returning from GitHub. Taking you to the dashboard…
    </p>
  );
}

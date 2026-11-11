/** @type {import('next').NextConfig} */

// Where the FastAPI backend runs during local development.
const apiBase = (process.env.API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');

const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  async rewrites() {
    // On Vercel, vercel.json sends /api/* straight to the Python function
    // before Next.js sees the request. A proxy rule would only add a route to
    // localhost to the deployed routing table, so none is emitted there.
    if (process.env.VERCEL && !process.env.API_BASE_URL) return [];
    // Local development (and self-hosting the web app next to the API).
    return [
      {
        source: '/api/:path*',
        destination: `${apiBase}/api/:path*`,
      },
    ];
  },
  experimental: {
    // A task streams its progress over a single request that stays open for
    // several minutes; the default proxy timeout is 30 seconds.
    proxyTimeout: 330_000,
  },
};

export default nextConfig;

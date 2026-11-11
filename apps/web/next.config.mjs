/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  // Support both local dev proxy and Vercel serverless rewrites
  async rewrites() {
    return process.env.NODE_ENV === 'development'
      ? [
          {
            source: '/api/v1/:path*',
            destination: 'http://localhost:8000/api/v1/:path*',
          },
        ]
      : [];
  },
};

export default nextConfig;

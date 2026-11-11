/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  async rewrites() {
    return [
      {
        source: '/api/v1/:path*',
        destination: process.env.API_BASE_URL
          ? `${process.env.API_BASE_URL}/api/v1/:path*`
          : 'http://localhost:8000/api/v1/:path*',
      },
    ];
  },
};

export default nextConfig;

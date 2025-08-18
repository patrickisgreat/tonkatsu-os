/** @type {import('next').NextConfig} */
const nextConfig = {
  // experimental: {
  //   appDir: true, // No longer needed in Next.js 14
  // },
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: 'http://127.0.0.1:8000/api/:path*', // FastAPI backend
      },
    ]
  },
  webpack: (config) => {
    // Handle plotly.js
    config.resolve.alias = {
      ...config.resolve.alias,
      'plotly.js': 'plotly.js/dist/plotly.min.js',
    }
    return config
  },
}

module.exports = nextConfig
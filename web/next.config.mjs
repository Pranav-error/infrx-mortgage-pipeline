/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config) => {
    // Required for react-pdf canvas support
    config.resolve.alias.canvas = false
    return config
  },
}

export default nextConfig

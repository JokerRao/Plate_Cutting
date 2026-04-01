export const API_CONFIG = {
  get BASE_URL() {
    if (typeof window !== 'undefined') {
      // If NEXT_PUBLIC_API_URL is set to something other than localhost, use it directly (e.g. production)
      const envUrl = process.env.NEXT_PUBLIC_API_URL;
      if (envUrl && !envUrl.includes('localhost') && !envUrl.includes('127.0.0.1')) {
        return envUrl;
      }
      // If we are accessing via LAN IP, automatically point to port 8000 on the same IP
      if (window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1') {
        return `${window.location.protocol}//${window.location.hostname}:8000`;
      }
    }
    return process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
  },
  POLLING_INTERVAL_MS: 10000,
  ENDPOINTS: {
    OPTIMIZE: '/optimize',
    OPTIMIZE_ASYNC: '/optimize/async',
    OPTIMIZE_JOB: '/optimize/jobs',
  },
} as const;

export const getApiUrl = (endpoint: keyof typeof API_CONFIG.ENDPOINTS) => {
  return `${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS[endpoint]}`;
}; 
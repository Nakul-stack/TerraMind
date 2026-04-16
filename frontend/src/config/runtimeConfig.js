const rawApiBaseUrl = import.meta.env.VITE_API_BASE_URL || '';

const trimTrailingSlash = (value) => value.replace(/\/+$/, '');

export const API_BASE_URL = trimTrailingSlash(rawApiBaseUrl);
export const API_V1_BASE_URL = `${API_BASE_URL}/api/v1`;

export const TIMEOUTS = {
  monitorRequestMs: Number(import.meta.env.VITE_MONITOR_TIMEOUT_MS || 30000),
};

export const DIAGNOSIS_DEFAULT_TOP_K = Number(
  import.meta.env.VITE_DIAGNOSIS_DEFAULT_TOP_K || 3
);

export const DIAGNOSIS_TOPK_OPTIONS = [1, 3, 5];

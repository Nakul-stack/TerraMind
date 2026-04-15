import { API_V1_BASE_URL } from '../config/runtimeConfig';

const API_BASE_URL = API_V1_BASE_URL;

export const graphRagService = {
  query: async (query, useLlm = true) => {
    const response = await fetch(`${API_BASE_URL}/graph-rag/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, use_llm: useLlm }),
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `AugNosis request failed (${response.status})`);
    }

    return await response.json();
  },

  health: async () => {
    const response = await fetch(`${API_BASE_URL}/graph-rag/health`);
    if (!response.ok) {
      throw new Error(`Health check failed (${response.status})`);
    }
    return await response.json();
  },
};

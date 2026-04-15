import { API_V1_BASE_URL } from '../config/runtimeConfig';

const API_BASE_URL = API_V1_BASE_URL;

export const advisorService = {
  /**
   * Run the unified pre-sowing advisory pipeline.
   * Returns: crop recommendation (top-3), yield prediction, irrigation advisory,
   *          district intelligence, and system notes.
   */
  predictBeforeSowing: async (data) => {
    try {
      const response = await fetch(`${API_BASE_URL}/advisor/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error("API Error in predictBeforeSowing:", error);
      throw error;
    }
  },

  /**
   * Train all 3 models.
   */
  trainAllModels: async () => {
    const response = await fetch(`${API_BASE_URL}/advisor/train/all`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || 'Training failed');
    }
    return await response.json();
  },

  /**
   * Get model metadata.
   */
  getMetadata: async () => {
    const response = await fetch(`${API_BASE_URL}/advisor/metadata`);
    if (!response.ok) throw new Error('Failed to fetch metadata');
    return await response.json();
  },
};

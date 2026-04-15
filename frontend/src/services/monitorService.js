import { API_V1_BASE_URL, TIMEOUTS } from '../config/runtimeConfig';

const API_BASE_URL = API_V1_BASE_URL;

export const monitorService = {
  /**
   * Predict active growth stage advisory (Model 2).
   */
  predictDuringGrowth: async (data) => {
    const controller = new AbortController();
    const timeoutMs = TIMEOUTS.monitorRequestMs;
    const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

    try {
      const response = await fetch(`${API_BASE_URL}/monitor/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        signal: controller.signal,
        body: JSON.stringify(data),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to fetch recommendation.');
      }

      return await response.json();
    } catch (error) {
      if (error.name === 'AbortError') {
        throw new Error('Monitor request timed out. Please retry in a few seconds.');
      }
      console.error("API Error in predictDuringGrowth:", error);
      throw error;
    } finally {
      clearTimeout(timeoutId);
    }
  }
};

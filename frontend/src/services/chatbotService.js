import { API_V1_BASE_URL } from '../config/runtimeConfig';

const API_BASE_URL = API_V1_BASE_URL;

export const chatbotService = {
  /**
   * Send a question to the PDF-grounded chatbot.
   * @param {string} question
   * @param {number} topK
   * @param {string|null} identifiedCrop
   * @param {string|null} identifiedClass
   * @param {string|null} reportId
   * @returns {Promise<object>}
   */
  askQuestion: async (question, topK = 5, identifiedCrop = null, identifiedClass = null, reportId = null) => {
    try {
      const body = {
        question,
        top_k: topK,
      };
      if (identifiedCrop) body.identified_crop = identifiedCrop;
      if (identifiedClass) body.identified_class = identifiedClass;
      if (reportId) body.report_id = reportId;

      const response = await fetch(`${API_BASE_URL}/chatbot/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${response.status})`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Error in askQuestion:', error);
      throw error;
    }
  },

  /**
   * Check chatbot system status (index loaded, Ollama available).
   * @returns {Promise<object>}
   */
  getStatus: async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/chatbot/status`);
      if (!response.ok) {
        throw new Error(`Status check failed (${response.status})`);
      }
      return await response.json();
    } catch (error) {
      console.error('API Error in getStatus:', error);
      throw error;
    }
  },
};

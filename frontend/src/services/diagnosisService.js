import { API_V1_BASE_URL } from '../config/runtimeConfig';

const API_BASE_URL = API_V1_BASE_URL;

export const diagnosisService = {
  /**
   * Upload a leaf/plant image for disease diagnosis.
   *
   * @param {File} imageFile - The image file to upload.
   * @param {number} topK - Number of top predictions to return.
   * @returns {Promise<object>} DiagnosisResponse JSON (includes report_id).
   */
  predictDiagnosis: async (imageFile, topK = 3) => {
    const formData = new FormData();
    formData.append('file', imageFile);

    const url = `${API_BASE_URL}/diagnosis/predict?top_k=${topK}`;

    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
        // Do NOT set Content-Type — browser sets it with the boundary for FormData
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || `Server error (${response.status})`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Error in predictDiagnosis:', error);
      throw error;
    }
  },

  /**
   * Poll the status of a background LLM diagnosis report.
   *
   * @param {string} reportId - The report tracking UUID returned by predictDiagnosis.
   * @returns {Promise<object>} Report status object: { status, data?, message? }
   */
  fetchReportStatus: async (reportId) => {
    const url = `${API_BASE_URL}/diagnosis/report/${reportId}`;

    try {
      const response = await fetch(url);

      if (response.status === 404) {
        return { status: 'not_found' };
      }

      if (!response.ok) {
        throw new Error(`Server error (${response.status})`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Error in fetchReportStatus:', error);
      throw error;
    }
  },

  /**
   * Mark a diagnosis report as downloaded on the backend to unlock TerraBot.
   *
   * @param {string} reportId - The report ID to mark as downloaded.
   * @returns {Promise<object>} Status object
   */
  markReportDownloaded: async (reportId) => {
    const url = `${API_BASE_URL}/diagnosis/report/${reportId}/download`;

    try {
      const response = await fetch(url, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`Server error (${response.status})`);
      }

      return await response.json();
    } catch (error) {
      console.error('API Error in markReportDownloaded:', error);
      throw error;
    }
  },
};

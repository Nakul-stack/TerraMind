import React, { useState, useEffect, useRef, useCallback } from 'react';
import ImageUploadForm from '../components/forms/ImageUploadForm';
import DiagnosisResultCard from '../components/results/DiagnosisResultCard';
import SmartAssistantDrawer from '../components/assistant/SmartAssistantDrawer';
import { diagnosisService } from '../services/diagnosisService';
import { chatbotService } from '../services/chatbotService';
import { generatePDF } from '../utils/generateReport';
import { Search, ShieldAlert, Zap, BookOpen } from 'lucide-react';

const POLL_INTERVAL_MS = 3000;
const POLL_TIMEOUT_MS = 240000;

const PROGRESS_MESSAGES = [
  [0, 'Initializing AugNosis extraction engine...'],
  [15, 'Traversing AgroKG for disease-specific context...'],
  [30, 'Synthesizing knowledge graph with LLM report generator...'],
  [60, 'Finalizing expert diagnostic report and treatment plan...'],
];

export default function Diagnosis() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Report polling state
  const [reportId, setReportId] = useState(null);
  const [reportStatus, setReportStatus] = useState(null); // "polling"|"ready"|"error"|"timeout"|null
  const [reportData, setReportData] = useState(null);
  const [reportReady, setReportReady] = useState(false);
  const [reportErrorMessage, setReportErrorMessage] = useState(null);
  const [hasDownloadedReport, setHasDownloadedReport] = useState(false);
  const [progressMessage, setProgressMessage] = useState(PROGRESS_MESSAGES[0][1]);
  const [assistantVisible, setAssistantVisible] = useState(false);
  const [assistantMessages, setAssistantMessages] = useState([]);
  const [assistantLoading, setAssistantLoading] = useState(false);
  const [assistantError, setAssistantError] = useState(null);

  const pollIntervalRef = useRef(null);
  const pollStartRef = useRef(null);
  const progressTimerRef = useRef(null);

  const stopPolling = useCallback(() => {
    if (pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }
    if (progressTimerRef.current) {
      clearInterval(progressTimerRef.current);
      progressTimerRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const startPolling = useCallback(
    (rid) => {
      stopPolling();
      setReportStatus('polling');
      setReportReady(false);
      setReportErrorMessage(null);
      setHasDownloadedReport(false);
      setReportData(null);
      setProgressMessage(PROGRESS_MESSAGES[0][1]);
      pollStartRef.current = Date.now();

      progressTimerRef.current = setInterval(() => {
        const elapsed = (Date.now() - pollStartRef.current) / 1000;
        for (let i = PROGRESS_MESSAGES.length - 1; i >= 0; i--) {
          if (elapsed >= PROGRESS_MESSAGES[i][0]) {
            setProgressMessage(PROGRESS_MESSAGES[i][1]);
            break;
          }
        }
      }, 1000);

      pollIntervalRef.current = setInterval(async () => {
        if (Date.now() - pollStartRef.current > POLL_TIMEOUT_MS) {
          stopPolling();
          setReportStatus('timeout');
          setReportErrorMessage('Report generation is taking longer than expected. Please try diagnosis again or verify Ollama/model availability.');
          return;
        }

        try {
          const res = await diagnosisService.fetchReportStatus(rid);

          if (res.status === 'ready') {
            stopPolling();
            setReportStatus('ready');
            setReportReady(true);
            setReportErrorMessage(null);
            setReportData(res.data);
          } else if (res.status === 'error') {
            stopPolling();
            setReportStatus('error');
            setReportErrorMessage(res.message || 'Report engine failed to generate the expert report.');
          } else if (res.status === 'not_found') {
            stopPolling();
            setReportStatus('error');
            setReportErrorMessage('Report session was not found. It may have expired; run diagnosis again.');
          } else if (res.status !== 'processing') {
            stopPolling();
            setReportStatus('error');
            setReportErrorMessage('Unexpected report status received from server.');
          }
        } catch (err) {
          console.warn('Report poll error:', err);
        }
      }, POLL_INTERVAL_MS);
    },
    [stopPolling]
  );

  const handleSubmit = async (imageFile, topK) => {
    setIsLoading(true);
    setError(null);
    setResult(null);
    setReportId(null);
    setReportStatus(null);
    setReportReady(false);
    setReportErrorMessage(null);
    setHasDownloadedReport(false);
    setReportData(null);
    setAssistantVisible(false);
    setAssistantMessages([]);
    setAssistantError(null);
    stopPolling();

    try {
      const data = await diagnosisService.predictDiagnosis(imageFile, topK);
      setResult(data);

      if (data.report_id) {
        setReportId(data.report_id);
        startPolling(data.report_id);
      }
    } catch (err) {
      setError(err.message || 'An unexpected error occurred during diagnosis.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleDownloadReport = useCallback(async () => {
    if (!reportData || !result) return;

    try {
      generatePDF(reportData, result.identified_crop, result.identified_class);
      if (reportId) {
        await diagnosisService.markReportDownloaded(reportId);
        setHasDownloadedReport(true);
      }
    } catch (err) {
      console.error('Report download error:', err);
    }
  }, [reportData, result, reportId]);

  const diagnosisComplete = Boolean(result);
  const reportDownloaded = Boolean(hasDownloadedReport);
  const assistantEnabled = Boolean(diagnosisComplete && reportReady && reportDownloaded);

  const assistantContext = {
    crop: result?.identified_crop || null,
    disease: result?.identified_class || null,
    confidencePct: result?.confidence ? (result.confidence * 100).toFixed(1) : null,
    reportId,
    reportData,
  };

  const handleOpenAssistant = useCallback(() => {
    if (!assistantEnabled) return;
    setAssistantVisible(true);
    setAssistantError(null);
  }, [assistantEnabled]);

  const handleAssistantClose = useCallback(() => {
    setAssistantVisible(false);
  }, []);

  const handleAssistantClear = useCallback(() => {
    setAssistantMessages([]);
    setAssistantError(null);
  }, []);

  const handleAssistantSend = useCallback(
    async (question) => {
      if (!assistantEnabled || !question?.trim()) return;

      setAssistantError(null);
      setAssistantMessages((prev) => [...prev, { role: 'user', text: question.trim() }]);
      setAssistantLoading(true);

      try {
        const data = await chatbotService.askQuestion(
          question.trim(),
          5,
          result?.identified_crop || null,
          result?.identified_class || null,
          reportId || null
        );

        const assistantMsg = {
          role: 'assistant',
          text: data.answer,
          allowed: data.allowed,
          reason: data.reason,
          sources: data.sources || [],
        };
        setAssistantMessages((prev) => [...prev, assistantMsg]);
      } catch (err) {
        setAssistantError(err.message || 'Failed to get recommendations from Smart Assistant.');
      } finally {
        setAssistantLoading(false);
      }
    },
    [assistantEnabled, reportId, result]
  );

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Page Header */}
      <div className="mb-10">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-rose-500 to-amber-600 flex items-center justify-center shadow-lg shadow-rose-500/20">
              <Search className="w-8 h-8 text-white" />
            </div>
            <div>
              <h2 className="text-3xl font-extrabold text-white tracking-tight">Post-Symptom Diagnosis</h2>
              <p className="text-sm text-slate-400 mt-1 max-w-xl leading-relaxed">
                Visual pathology recognition using deep CNN ensembles. 
                Integrated with AugNosis for expert-level diagnostic report generation.
              </p>
            </div>
          </div>
          
          <div className="flex gap-2">
            {[
              { icon: Zap, label: 'AugNosis Enabled', color: 'sky' },
              { icon: BookOpen, label: 'Expert Verified', color: 'emerald' }
            ].map((feature) => (
              <div key={feature.label} className={`flex items-center gap-2 px-4 py-2 rounded-xl bg-${feature.color}-500/5 border border-${feature.color}-500/10`}>
                <feature.icon className={`w-3.5 h-3.5 text-${feature.color}-400`} />
                <span className="text-[10px] font-bold text-white uppercase tracking-widest">{feature.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Intelligence Banner */}
        <div className="mt-8 glass-card border-slate-800/50 bg-slate-800/20 p-4 border-l-4 border-l-rose-500/50">
          <div className="flex items-start gap-3">
            <ShieldAlert className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
            <p className="text-xs text-slate-300 leading-relaxed">
              Upload high-resolution leaf imagery for maximum precision. The pipeline will first identify the <span className="text-emerald-400 font-bold">host crop</span>, 
              then run symptom classification, and finally query the <span className="text-sky-400 font-bold">Agro Knowledge Graph</span> to synthesize a tailored treatment plan.
            </p>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto space-y-8 pb-12">
        <ImageUploadForm onSubmit={handleSubmit} isLoading={isLoading} />

        {error && (
          <div className="glass-card p-6 border-rose-500/20 bg-rose-500/5 animate-in shake duration-500">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-xl bg-rose-500/10 flex items-center justify-center shrink-0">
                <ShieldAlert className="w-5 h-5 text-rose-400" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-rose-400 mb-1">Diagnostic Engine Failure</h3>
                <p className="text-sm text-slate-400">{error}</p>
              </div>
            </div>
          </div>
        )}

        {result && (
          <DiagnosisResultCard
            result={result}
            reportId={reportId}
            reportStatus={reportStatus}
            reportReady={reportReady}
            reportErrorMessage={reportErrorMessage}
            hasDownloadedReport={hasDownloadedReport}
            progressMessage={progressMessage}
            onDownloadReport={handleDownloadReport}
            onOpenAssistant={handleOpenAssistant}
            assistantEnabled={assistantEnabled}
          />
        )}

        {!result && !isLoading && !error && (
            <div className="py-20 flex flex-col items-center justify-center opacity-40 grayscale min-h-[300px]">
                <Search className="w-16 h-16 text-slate-600 mb-4 animate-pulse" />
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.3em]">Awaiting Visual Input Stream</p>
            </div>
        )}
      </div>

      <SmartAssistantDrawer
        isOpen={assistantVisible}
        onClose={handleAssistantClose}
        onSend={handleAssistantSend}
        onClear={handleAssistantClear}
        messages={assistantMessages}
        isLoading={assistantLoading}
        error={assistantError}
        assistantContext={assistantContext}
      />
    </div>
  );
}

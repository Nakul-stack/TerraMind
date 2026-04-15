import React, { useEffect, useMemo, useRef, useState } from 'react';
import {
  X,
  Sparkles,
  Send,
  ShieldCheck,
  Leaf,
  Bug,
  Gauge,
  Loader2,
  AlertCircle,
  MessageSquare,
  ListChecks,
  FileText,
  RefreshCw,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import ChatMessageCard from '../results/ChatMessageCard';

function formatDisease(rawDisease) {
  if (!rawDisease) return 'Unknown disease';
  const parts = rawDisease.split('__');
  const disease = parts.length > 1 ? parts[1] : parts[0];
  return disease.replace(/_/g, ' ');
}

function normalizeText(value) {
  if (!value) return 'Not available';
  const normalized = String(value).replace(/\s+/g, ' ').trim();
  if (!normalized) return 'Not available';
  return normalized;
}

export default function SmartAssistantDrawer({
  isOpen,
  onClose,
  onSend,
  onClear,
  messages,
  isLoading,
  error,
  assistantContext,
}) {
  const [question, setQuestion] = useState('');
  const [isContextExpanded, setIsContextExpanded] = useState(false);
  const [isReportExpanded, setIsReportExpanded] = useState(false);
  const messagesEndRef = useRef(null);

  useEffect(() => {
    if (!isOpen) return;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = '';
    };
  }, [isOpen]);

  useEffect(() => {
    if (isOpen) {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isLoading, isOpen]);

  const contextSummary = useMemo(() => {
    const report = assistantContext?.reportData || {};
    return {
      crop: assistantContext?.crop || 'Unknown crop',
      disease: formatDisease(assistantContext?.disease),
      confidence: assistantContext?.confidencePct || 'N/A',
      reportId: assistantContext?.reportId || 'N/A',
      diseaseOverview: normalizeText(report.disease_overview),
      immediateSteps: normalizeText(report.immediate_steps),
      treatment: normalizeText(report.treatment),
      prevention: normalizeText(report.prevention),
      monitoringAdvice: normalizeText(report.monitoring_advice),
      severity: normalizeText(report.severity),
    };
  }, [assistantContext]);

  const submitQuestion = (e) => {
    e.preventDefault();
    const cleaned = question.trim();
    if (!cleaned || isLoading) return;
    onSend(cleaned);
    setQuestion('');
  };

  return (
    <>
      <div
        className={`fixed inset-0 z-[70] bg-slate-950/60 backdrop-blur-sm transition-opacity duration-300 ${
          isOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
        onClick={onClose}
        aria-hidden="true"
      />

      <aside
        className={`fixed right-0 top-0 z-[80] h-full w-full sm:w-[92vw] lg:w-[540px] xl:w-[620px] border-l border-sky-500/20 bg-slate-950/95 shadow-2xl shadow-black/50 transition-transform duration-500 ease-out flex flex-col overflow-hidden ${
          isOpen ? 'translate-x-0' : 'translate-x-full'
        }`}
        role="dialog"
        aria-modal="true"
        aria-label="Smart assistant panel"
      >
        <div className="px-5 sm:px-6 py-4 border-b border-slate-800/80 bg-gradient-to-r from-sky-500/10 via-transparent to-transparent">
          <div className="flex items-start justify-between gap-4">
            <div className="flex items-start gap-3">
              <div className="w-11 h-11 rounded-2xl bg-sky-500/15 border border-sky-500/30 flex items-center justify-center text-sky-300">
                <Sparkles className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base sm:text-lg font-black text-white tracking-tight">
                  Smart Assistant Recommendations
                </h3>
                <p className="text-[11px] text-slate-400 mt-1 leading-relaxed">
                  Continuing from your diagnosis report with crop-specific treatment, prevention, and next-step guidance.
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={onClose}
              className="p-2 rounded-xl border border-slate-700/60 text-slate-400 hover:text-white hover:border-slate-500 transition-colors"
              aria-label="Close assistant panel"
            >
              <X className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="flex-1 min-h-0 flex flex-col bg-slate-950/35">
          <div className="flex-1 min-h-0 overflow-y-auto p-5 sm:p-6 custom-scrollbar">
            <div className="mb-4 space-y-3">
              <div className="rounded-2xl border border-emerald-500/25 bg-emerald-500/5">
                <button
                  type="button"
                  onClick={() => setIsContextExpanded((prev) => !prev)}
                  className="w-full p-4 text-left flex items-center justify-between gap-3"
                  aria-expanded={isContextExpanded}
                >
                  <div className="min-w-0">
                    <p className="text-[10px] font-bold text-emerald-400 uppercase tracking-[0.18em] flex items-center gap-2">
                      <ShieldCheck className="w-3.5 h-3.5" /> Diagnosis Context
                    </p>
                    <p className="text-xs text-slate-300 mt-1 truncate">
                      {contextSummary.crop} | {contextSummary.disease} | {contextSummary.confidence}% confidence
                    </p>
                  </div>
                  {isContextExpanded ? (
                    <ChevronUp className="w-4 h-4 text-slate-400 shrink-0" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
                  )}
                </button>

                {isContextExpanded && (
                  <div className="px-4 pb-4">
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                      <div className="rounded-xl border border-slate-800 p-3 bg-slate-900/70">
                        <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 flex items-center gap-1">
                          <Leaf className="w-3 h-3" /> Crop
                        </p>
                        <p className="text-white font-bold">{contextSummary.crop}</p>
                      </div>
                      <div className="rounded-xl border border-slate-800 p-3 bg-slate-900/70">
                        <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 flex items-center gap-1">
                          <Bug className="w-3 h-3" /> Disease
                        </p>
                        <p className="text-rose-300 font-bold capitalize">{contextSummary.disease}</p>
                      </div>
                      <div className="rounded-xl border border-slate-800 p-3 bg-slate-900/70">
                        <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 flex items-center gap-1">
                          <Gauge className="w-3 h-3" /> Confidence
                        </p>
                        <p className="text-sky-300 font-bold">{contextSummary.confidence}%</p>
                      </div>
                      <div className="rounded-xl border border-slate-800 p-3 bg-slate-900/70">
                        <p className="text-[10px] text-slate-500 uppercase tracking-widest mb-1 flex items-center gap-1">
                          <FileText className="w-3 h-3" /> Report Session
                        </p>
                        <p className="text-white font-semibold truncate" title={contextSummary.reportId}>{contextSummary.reportId}</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              <div className="rounded-2xl border border-slate-800/70 bg-slate-900/60">
                <button
                  type="button"
                  onClick={() => setIsReportExpanded((prev) => !prev)}
                  className="w-full p-4 text-left flex items-center justify-between gap-3"
                  aria-expanded={isReportExpanded}
                >
                  <div className="min-w-0">
                    <p className="text-[10px] text-slate-400 uppercase tracking-widest font-bold flex items-center gap-2">
                      <ListChecks className="w-3.5 h-3.5" /> Report Summary Snapshot
                    </p>
                    <p className="text-xs text-slate-300 mt-1 truncate">
                      Severity: {contextSummary.severity}
                    </p>
                  </div>
                  {isReportExpanded ? (
                    <ChevronUp className="w-4 h-4 text-slate-400 shrink-0" />
                  ) : (
                    <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
                  )}
                </button>

                {isReportExpanded && (
                  <div className="px-4 pb-4 space-y-3 text-[12px] text-slate-300 leading-relaxed">
                    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
                      <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Immediate Steps</p>
                      <p>{contextSummary.immediateSteps}</p>
                    </div>
                    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
                      <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Treatment</p>
                      <p>{contextSummary.treatment}</p>
                    </div>
                    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
                      <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Prevention</p>
                      <p>{contextSummary.prevention}</p>
                    </div>
                    <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-3">
                      <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Monitoring Advice</p>
                      <p>{contextSummary.monitoringAdvice}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

          {messages.length === 0 && !isLoading && (
            <div className="h-full min-h-[220px] rounded-2xl border border-dashed border-slate-700/80 bg-slate-900/30 flex flex-col items-center justify-center text-center p-6">
              <MessageSquare className="w-10 h-10 text-slate-600 mb-4" />
              <h4 className="text-sm font-black text-white uppercase tracking-wider mb-2">Ask for Specific Recommendations</h4>
              <p className="text-xs text-slate-400 leading-relaxed max-w-sm">
                Ask about pesticides, treatment sequence, prevention strategy, or 7-day action plan. The assistant uses your active diagnosis and report context.
              </p>
            </div>
          )}

          {messages.map((message, idx) => (
            <ChatMessageCard key={idx} message={message} />
          ))}

          {isLoading && (
            <div className="flex gap-3 items-center p-3 rounded-xl bg-slate-900/70 border border-slate-800 w-fit">
              <Loader2 className="w-4 h-4 text-sky-400 animate-spin" />
              <span className="text-[11px] text-slate-300 uppercase tracking-widest font-bold">Generating recommendations...</span>
            </div>
          )}

          {error && (
            <div className="mt-4 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex gap-2 items-start">
              <AlertCircle className="w-4 h-4 mt-0.5" />
              <span>{error}</span>
            </div>
          )}

          <div ref={messagesEndRef} />
          </div>
        </div>

        <div className="shrink-0 border-t border-slate-800/80 p-4 sm:p-5 bg-slate-900/80 backdrop-blur-xl">
          <form onSubmit={submitQuestion} className="flex items-center gap-3">
            <input
              type="text"
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Ask treatment, prevention, pesticide, or next-step advice..."
              disabled={isLoading}
              className="flex-1 bg-slate-950/80 border border-slate-700/60 rounded-2xl px-4 py-3 text-sm text-white focus:ring-2 focus:ring-sky-500/30 focus:border-sky-500 outline-none placeholder:text-slate-600"
            />
            <button
              type="submit"
              disabled={!question.trim() || isLoading}
              className="px-4 py-3 rounded-2xl bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-600 text-white transition-colors font-bold text-xs uppercase tracking-widest inline-flex items-center gap-2"
            >
              <Send className="w-4 h-4" /> Send
            </button>
            {messages.length > 0 && (
              <button
                type="button"
                onClick={onClear}
                className="px-3 py-3 rounded-2xl border border-slate-700 text-slate-400 hover:text-white hover:border-slate-500 transition-colors"
                title="Clear assistant chat"
                aria-label="Clear assistant chat"
              >
                <RefreshCw className="w-4 h-4" />
              </button>
            )}
          </form>
        </div>
      </aside>
    </>
  );
}

import React from 'react';
import { Leaf, Bug, BarChart3, MessageCircle, Download, Loader2, Award, Zap, ShieldCheck, AlertCircle } from 'lucide-react';

export default function DiagnosisResultCard({
  result,
  reportId,
  reportStatus,
  reportReady,
  reportErrorMessage,
  hasDownloadedReport,
  progressMessage,
  onDownloadReport,
  onOpenAssistant,
  assistantEnabled,
}) {

  if (!result) return null;

  const confidencePct = (result.confidence * 100).toFixed(1);

  const formatDisease = (cls) => {
    if (!cls) return 'Unknown';
    const parts = cls.split('__');
    const disease = parts.length > 1 ? parts[1] : parts[0];
    return disease.replace(/_/g, ' ');
  };

  return (
    <div className="mt-12 animate-in fade-in slide-in-from-bottom-6 duration-700">
      {/* Header section with summary badge */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 mb-8">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-500 to-rose-600 flex items-center justify-center shadow-lg shadow-rose-500/20">
            <Bug className="w-7 h-7 text-white" />
          </div>
          <div>
            <h3 className="text-2xl font-extrabold text-white tracking-tight">Diagnosis Report</h3>
            <div className="flex items-center gap-2 mt-1">
              <span className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Neural Network identified pathology</span>
              <div className="w-1 h-1 rounded-full bg-slate-700" />
              <span className="text-[10px] text-emerald-500 font-bold uppercase tracking-widest">High Reliability</span>
            </div>
          </div>
        </div>
        
        <div className="px-6 py-2 rounded-2xl bg-slate-800/80 border border-slate-700/50 backdrop-blur-md flex items-center gap-3">
          <div className="text-right">
            <p className="text-[9px] font-bold text-slate-500 uppercase tracking-tighter">Overall Engine</p>
            <p className="text-xs font-bold text-white uppercase">Confidence</p>
          </div>
          <div className="h-8 w-[1px] bg-slate-700" />
          <div className="text-3xl font-black text-emerald-400 font-mono tracking-tighter">
            {confidencePct}<span className="text-sm ml-0.5">%</span>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Left Column: Primary Findings */}
        <div className="lg:col-span-8 space-y-8">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
             {/* Identified Crop */}
            <div className="glass-card p-6 border-emerald-500/20 bg-emerald-500/5 relative overflow-hidden group">
              <div className="absolute -right-4 -top-4 w-24 h-24 bg-emerald-500/5 blur-3xl rounded-full" />
              <div className="flex items-start gap-4 h-full">
                <div className="p-3 bg-emerald-500/10 rounded-2xl text-emerald-400 group-hover:scale-110 transition-transform">
                  <Leaf className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Target Crop</p>
                  <p className="text-xl font-black text-white">{result.identified_crop}</p>
                </div>
              </div>
            </div>

            {/* Disease Detected */}
            <div className="glass-card p-6 border-rose-500/20 bg-rose-500/5 relative overflow-hidden group">
              <div className="absolute -right-4 -top-4 w-24 h-24 bg-rose-500/5 blur-3xl rounded-full" />
              <div className="flex items-start gap-4">
                <div className="p-3 bg-rose-500/10 rounded-2xl text-rose-400 group-hover:scale-110 transition-transform">
                  <Bug className="w-6 h-6" />
                </div>
                <div>
                  <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">Detected Pathology</p>
                  <p className="text-xl font-black text-white capitalize">{formatDisease(result.identified_class)}</p>
                  <p className="text-[10px] text-rose-400/50 mt-1 font-mono uppercase tracking-tighter">{result.identified_class}</p>
                </div>
              </div>
            </div>
          </div>

          {/* Top-K Predictions Table */}
          {result.top_k_predictions && result.top_k_predictions.length > 0 && (
            <div className="glass-card overflow-hidden border-slate-800/50">
              <div className="bg-slate-800/20 p-4 border-b border-slate-800/50 flex items-center justify-between">
                <h4 className="text-[10px] font-extrabold text-white uppercase tracking-widest flex items-center gap-2">
                  <BarChart3 className="w-3.5 h-3.5 text-emerald-500" />
                  Neural Prediction Ranking
                </h4>
                <div className="text-[9px] text-slate-500 font-bold uppercase tracking-widest">
                  Software v1.22
                </div>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm text-left">
                  <thead>
                    <tr className="border-b border-slate-800 text-[9px] text-slate-500 font-bold uppercase tracking-widest bg-slate-900/40">
                      <th className="px-6 py-4">Rank</th>
                      <th className="px-6 py-4">Identification Path</th>
                      <th className="px-6 py-4 text-right">Probability Score</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800/50">
                    {result.top_k_predictions.map((pred, idx) => (
                      <tr key={idx} className={`group hover:bg-slate-800/30 transition-colors ${idx === 0 ? 'bg-emerald-500/5' : ''}`}>
                        <td className="px-6 py-4 font-mono text-[10px] text-slate-500 group-hover:text-white transition-colors">#{idx + 1}</td>
                        <td className="px-6 py-4">
                          <div className="flex flex-col">
                            <span className="text-white font-bold">{formatDisease(pred.class)}</span>
                            <span className="text-[10px] text-slate-500 leading-none mt-1">{pred.crop}</span>
                          </div>
                        </td>
                        <td className="px-6 py-4 text-right">
                          <div className="flex items-center justify-end gap-3">
                            <div className="hidden sm:block w-24 h-1 bg-slate-800 rounded-full overflow-hidden">
                              <div className={`h-full ${idx === 0 ? 'bg-emerald-500' : 'bg-slate-600'} rounded-full`} style={{ width: `${pred.confidence * 100}%` }} />
                            </div>
                            <span className={`font-mono font-bold ${idx === 0 ? 'text-emerald-400' : 'text-slate-400'}`}>
                              {(pred.confidence * 100).toFixed(2)}%
                            </span>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Expert Report & Assistant */}
        <div className="lg:col-span-4 space-y-6">
          <div className="glass-card p-6 border-sky-500/20 bg-sky-500/5 h-full flex flex-col">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2.5 bg-sky-500/10 rounded-xl text-sky-400">
                <ShieldCheck className="w-5 h-5" />
              </div>
              <div>
                <h4 className="text-sm font-extrabold text-white uppercase tracking-widest leading-none">Expert Validation</h4>
                <p className="text-[9px] text-slate-500 uppercase font-bold mt-1">AugNosis Pipeline</p>
              </div>
            </div>

            <div className="flex-1 space-y-6">
              {/* Dynamic Status Display */}
              <div className="p-4 bg-slate-900/50 rounded-2xl border border-slate-800 transition-all">
                {reportStatus === 'polling' && (
                  <div className="flex flex-col gap-3">
                    <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 text-sky-400 animate-spin" />
                        <span className="text-xs font-bold text-white uppercase tracking-tighter">Orchestrating Knowledge...</span>
                    </div>
                    <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                        <div className="h-full bg-sky-500 animate-[loading-bar_20s_infinite]" />
                    </div>
                    <p className="text-[10px] text-slate-400 leading-relaxed italic">"{progressMessage}"</p>
                  </div>
                )}

                {reportStatus === 'error' && (
                  <div className="flex flex-col gap-2 text-rose-400">
                    <div className="flex items-center gap-3">
                      <AlertCircle className="w-5 h-5 shrink-0" />
                      <span className="text-xs font-black uppercase">Report Engine Error</span>
                    </div>
                    {reportErrorMessage && (
                      <p className="text-[11px] text-rose-300/90 leading-relaxed">{reportErrorMessage}</p>
                    )}
                  </div>
                )}

                {reportStatus === 'timeout' && (
                  <div className="flex flex-col gap-2 text-amber-300">
                    <div className="flex items-center gap-3">
                      <AlertCircle className="w-5 h-5 shrink-0" />
                      <span className="text-xs font-black uppercase">Report Generation Timeout</span>
                    </div>
                    <p className="text-[11px] text-amber-200/90 leading-relaxed">
                      {reportErrorMessage || 'The report is taking too long. Check Ollama/model readiness and retry diagnosis.'}
                    </p>
                  </div>
                )}

                {reportReady && (
                  <div className="flex flex-col gap-4">
                    <div className="flex items-center gap-2">
                        <Award className="w-5 h-5 text-emerald-400" />
                        <span className="text-xs font-black text-white uppercase tracking-widest">Report Synthesized</span>
                    </div>
                    <p className="text-xs text-slate-400 leading-relaxed">
                      Knowledge graph extraction complete. AugNosis context was injected into the LLM prompt for precision.
                    </p>
                    {hasDownloadedReport && (
                        <div className="flex items-center gap-2 px-3 py-1.5 bg-emerald-500/10 border border-emerald-500/20 rounded-lg">
                            <Zap className="w-3 h-3 text-emerald-400" />
                            <span className="text-[9px] font-bold text-emerald-400 uppercase">Smart Assistant Ready</span>
                        </div>
                    )}
                  </div>
                )}

                {!reportStatus && !reportReady && (
                   <p className="text-xs text-slate-500 italic">Initiate diagnosis first to generate report...</p>
                )}

                {reportStatus && !['polling', 'error', 'timeout'].includes(reportStatus) && !reportReady && (
                  <p className="text-xs text-slate-400 italic">Waiting for report status update...</p>
                )}
              </div>

              {/* Action Buttons */}
              <div className="space-y-3">
                <button
                  type="button"
                  onClick={onDownloadReport}
                  disabled={!reportReady}
                  className={`w-full font-bold py-3.5 px-6 rounded-xl transition-all flex items-center justify-center gap-3 text-sm shadow-lg ${
                    reportReady
                      ? 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-500/20'
                      : 'bg-slate-800 text-slate-600 cursor-not-allowed border border-slate-700'
                  }`}
                >
                  <Download className="w-4 h-4" />
                  {hasDownloadedReport ? 'Download Again' : 'Download Expert PDF'}
                </button>

                {assistantEnabled && (
                  <button
                    type="button"
                    onClick={onOpenAssistant}
                    className="w-full bg-sky-600 hover:bg-sky-500 text-white font-bold py-3.5 px-6 rounded-xl transition-all flex items-center justify-center gap-3 text-sm shadow-lg shadow-sky-500/20 animate-in fade-in slide-in-from-top-4 duration-500"
                  >
                    <MessageCircle className="w-4 h-4" />
                    Ask Smart Assistant for Specific Recommendations
                  </button>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

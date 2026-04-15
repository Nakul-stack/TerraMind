import React, { useState, useEffect } from 'react';
import PredictionForm from '../components/PredictionForm';
import ResultsDashboard from '../components/ResultsDashboard';
import { predict } from '../services/api';
import { Leaf, Wifi, WifiOff, Zap, Sprout, Info } from 'lucide-react';

export default function Advisor() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (payload) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const data = await predict(payload);
      setResult(data);
    } catch (err) {
      setError(err.message || 'Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Header Info */}
      <div className="mb-8">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-12 h-12 rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
            <Sprout className="w-6 h-6 text-white" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white tracking-tight">Pre-Sowing Advisor</h2>
            <p className="text-sm text-slate-400">Powered by Hybrid Ensemble Models & District Intelligence</p>
          </div>
        </div>
        
        <div className="glass-card p-4 flex items-start gap-3 border-emerald-500/10 bg-emerald-500/5">
          <div className="w-8 h-8 rounded-full bg-emerald-500/10 flex items-center justify-center flex-shrink-0">
            <Info className="w-4 h-4 text-emerald-400" />
          </div>
          <p className="text-sm text-slate-300 leading-relaxed">
            Optimize your crop strategy by analyzing soil parameters, climatic forecast, and district historical trends. 
            Choose between <span className="text-emerald-400 font-bold">Central</span> (Full Power) or <span className="text-sky-400 font-bold">Edge</span> (Offline Optimized) modes.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Form Column */}
        <div className="lg:col-span-6 xl:col-span-5">
          <div className="lg:sticky lg:top-24">
            <PredictionForm onSubmit={handleSubmit} loading={loading} />
          </div>
        </div>

        {/* Results Column */}
        <div className="lg:col-span-6 xl:col-span-7">
          {loading && (
            <div className="glass-card min-h-[500px] flex flex-col items-center justify-center gap-6">
              <div className="relative">
                <div className="w-16 h-16 rounded-full border-4 border-emerald-500/10 border-t-emerald-500 animate-spin" />
                <Leaf className="w-6 h-6 text-emerald-500 absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-pulse" />
              </div>
              <div className="text-center">
                <p className="text-white font-bold text-lg mb-1">Synthesizing Advisory</p>
                <p className="text-slate-400 text-sm">Running ensemble inference & district priors synchronization...</p>
              </div>
            </div>
          )}

          {error && (
            <div className="glass-card p-8 border-rose-500/20 bg-rose-500/5">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 rounded-2xl bg-rose-500/10 flex items-center justify-center flex-shrink-0">
                  <WifiOff className="w-6 h-6 text-rose-400" />
                </div>
                <div>
                  <h3 className="text-xl font-bold text-rose-400 mb-2">Analysis Interrupted</h3>
                  <p className="text-slate-300 mb-4">{error}</p>
                  <div className="flex items-center gap-2 text-xs text-slate-500">
                    <span className="w-2 h-2 rounded-full bg-rose-500 animate-pulse" />
                    Check backend connection or model training status.
                  </div>
                </div>
              </div>
            </div>
          )}

          {result && <ResultsDashboard result={result} />}

          {!loading && !error && !result && (
            <div className="glass-card min-h-[500px] flex flex-col items-center justify-center gap-8 text-center p-12 overflow-hidden relative">
              <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-emerald-500 via-sky-500 to-emerald-500 opacity-20" />
              
              <div className="relative group">
                <div className="absolute inset-0 bg-emerald-500/20 blur-3xl rounded-full group-hover:bg-emerald-500/30 transition-all duration-500" />
                <div className="relative w-24 h-24 rounded-3xl bg-gradient-to-br from-emerald-500/20 to-lime-500/20 flex items-center justify-center border border-emerald-500/20 shadow-inner group-hover:scale-110 transition-transform duration-500">
                  <Leaf className="w-12 h-12 text-emerald-500" />
                </div>
              </div>

              <div className="max-w-md relative">
                <h3 className="text-2xl font-extrabold text-white mb-3 tracking-tight">Intelligence Ready</h3>
                <p className="text-slate-400 text-base leading-relaxed mb-8">
                  Fill in the agronomic profile on the left to generate real-time 
                  recommendations for <span className="text-emerald-400 font-semibold">crops</span>, <span className="text-sky-400 font-semibold">yields</span>, 
                  and <span className="text-amber-400 font-semibold">irrigation</span> efficiency.
                </p>
                
                <div className="flex flex-wrap justify-center gap-2">
                  {[
                    { label: '🌾 Crop Recos', color: 'emerald' },
                    { label: '📊 Yield Prediction', color: 'sky' },
                    { label: '🗺️ District Intel', color: 'violet' },
                    { label: '💧 Irrigation', color: 'amber' }
                  ].map((tag) => (
                    <span 
                      key={tag.label} 
                      className={`px-4 py-1.5 rounded-full bg-${tag.color}-500/10 text-[10px] font-bold text-${tag.color}-400 border border-${tag.color}-500/20 uppercase tracking-wider`}
                    >
                      {tag.label}
                    </span>
                  ))}
                </div>
              </div>

              {/* Decorative elements */}
              <div className="absolute top-10 right-10 w-32 h-32 bg-sky-500/5 blur-3xl rounded-full" />
              <div className="absolute bottom-10 left-10 w-32 h-32 bg-emerald-500/5 blur-3xl rounded-full" />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

import React, { useState } from 'react';
import GrowthStageForm from '../components/forms/GrowthStageForm';
import GrowthStageResultCard from '../components/results/GrowthStageResultCard';
import { monitorService } from '../services/monitorService';
import { Activity, Beaker, ShieldCheck, Zap } from 'lucide-react';

export default function Monitor() {
  const [result, setResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (formData) => {
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const data = await monitorService.predictDuringGrowth(formData);
      setResult(data);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred while processing the growth stage pipeline.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Page Header */}
      <div className="mb-10">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-emerald-500 to-sky-600 flex items-center justify-center shadow-lg shadow-emerald-500/20">
              <Activity className="w-8 h-8 text-white" />
            </div>
            <div>
              <h2 className="text-3xl font-extrabold text-white tracking-tight">Growth Stage Monitor</h2>
              <p className="text-sm text-slate-400 mt-1 max-w-xl leading-relaxed">
                Advanced causal ML pipeline for mid-growth lifecycle optimization. 
                Sequentially determines nutritional deficits, pest risks, and yield trajectories.
              </p>
            </div>
          </div>
          
          <div className="flex gap-2">
            {[
              { icon: Beaker, label: 'Causal Inference', color: 'emerald' },
              { icon: ShieldCheck, label: 'Pest Guard', color: 'sky' }
            ].map((feature) => (
              <div key={feature.label} className={`flex items-center gap-2 px-4 py-2 rounded-xl bg-${feature.color}-500/5 border border-${feature.color}-500/10`}>
                <feature.icon className={`w-3.5 h-3.5 text-${feature.color}-400`} />
                <span className="text-[10px] font-bold text-white uppercase tracking-widest">{feature.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Pipeline Info Banner */}
        <div className="mt-8 glass-card border-slate-800/50 bg-slate-800/20 p-4 border-l-4 border-l-emerald-500/50">
          <div className="flex items-start gap-3">
            <Zap className="w-5 h-5 text-emerald-400 shrink-0 mt-0.5" />
            <p className="text-xs text-slate-300 leading-relaxed">
              Monitoring is crucial during the active growth phase. Our system uses a multi-stage dependency parser to ensure that 
              <span className="text-emerald-400 font-bold"> fertilizer dosage</span> is perfectly timed with <span className="text-sky-400 font-bold">climatic stability</span> and <span className="text-rose-400 font-bold">pest thresholds</span>.
            </p>
          </div>
        </div>
      </div>

      <div className="max-w-6xl mx-auto space-y-8">
        <GrowthStageForm onSubmit={handleSubmit} isLoading={isLoading} />

        {error && (
          <div className="glass-card p-6 border-rose-500/20 bg-rose-500/5 animate-in shake duration-500">
            <div className="flex items-start gap-4">
              <div className="w-10 h-10 rounded-xl bg-rose-500/10 flex items-center justify-center shrink-0">
                <ShieldCheck className="w-5 h-5 text-rose-400" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-rose-400 mb-1">Pipeline Execution Failure</h3>
                <p className="text-sm text-slate-400">{error}</p>
              </div>
            </div>
          </div>
        )}

        {result && <GrowthStageResultCard result={result} />}
        
        {!result && !isLoading && !error && (
            <div className="py-20 flex flex-col items-center justify-center opacity-40 grayscale min-h-[300px]">
                <Activity className="w-16 h-16 text-slate-600 mb-4 animate-pulse" />
                <p className="text-[10px] font-bold text-slate-500 uppercase tracking-[0.3em]">Awaiting sensor telemetry input</p>
            </div>
        )}
      </div>
    </div>
  );
}

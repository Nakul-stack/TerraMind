import React from 'react';
import { Leaf, Droplets, Bug, Sprout, CalendarClock, TrendingUp, ShieldCheck, AlertCircle, Zap } from 'lucide-react';

const ResultItem = ({ icon: Icon, label, value, highlight = false, alert = false }) => (
  <div className={`glass-card p-5 relative overflow-hidden transition-all duration-500 group ${
    highlight ? 'border-emerald-500/20 bg-emerald-500/5' : 
    alert ? 'border-rose-500/20 bg-rose-500/5' : 
    'border-slate-800/50 hover:bg-slate-800/20'
  }`}>
    {/* Decorative blur */}
    <div className={`absolute -right-4 -top-4 w-12 h-12 blur-2xl rounded-full opacity-10 ${
      highlight ? 'bg-emerald-500' : alert ? 'bg-rose-500' : 'bg-sky-500'
    }`} />
    
    <div className="flex items-start gap-4 relative z-10">
      <div className={`p-3 rounded-2xl ${
        highlight ? 'bg-emerald-500/10 text-emerald-400 group-hover:scale-110 transition-transform' : 
        alert ? 'bg-rose-500/10 text-rose-400 animate-pulse' : 
        'bg-slate-800/80 text-sky-400'
      }`}>
        <Icon className="w-5 h-5" />
      </div>
      <div>
        <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mb-1">{label}</p>
        <p className={`text-lg font-extrabold tracking-tight ${
          highlight ? 'text-emerald-400' : alert ? 'text-rose-400' : 'text-white capitalize'
        }`}>
          {value}
        </p>
      </div>
    </div>
  </div>
);

export default function GrowthStageResultCard({ result }) {
  if (!result) return null;

  return (
    <div className="mt-12 animate-in fade-in slide-in-from-bottom-6 duration-700">
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4 mb-6">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
            <Sprout className="w-5 h-5 text-emerald-500" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-white tracking-tight">Advisory Execution</h3>
            <p className="text-[10px] text-slate-500 uppercase font-bold tracking-widest">Optimized Growth Protocol Generated</p>
          </div>
        </div>
        
        <div className="flex items-center gap-2 px-4 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
          <ShieldCheck className="w-3 h-3 text-emerald-400" />
          <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-widest">Causal Analysis Confirmed</span>
        </div>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <ResultItem 
          icon={Leaf}
          label="Recommended Fertilizer"
          value={result.recommended_fertilizer}
          highlight={true}
        />
        
        <ResultItem 
          icon={result.pest_level === 'High' ? AlertCircle : Bug}
          label="Pest Level Risk"
          value={result.pest_level}
          alert={result.pest_level === 'High'}
          highlight={result.pest_level === 'Medium'}
        />

        <ResultItem 
          icon={Droplets}
          label="Prescribed Dosage"
          value={`${result.dosage} kg/acre`}
        />
        
        <ResultItem 
          icon={CalendarClock}
          label="Application Timeline"
          value={`In ${result.apply_after_days} Days`}
          highlight={true}
        />
        
        <ResultItem 
          icon={TrendingUp}
          label="Projected Yield"
          value={`${result.expected_yield_after_dosage} q/ha`}
          highlight={true}
        />
      </div>

      {/* Advisory Insight Banner */}
      <div className="mt-8 p-6 glass-card border-sky-500/10 bg-sky-500/5 relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity">
          <Zap className="w-20 h-20 text-sky-400" />
        </div>
        <div className="flex items-start gap-4 relative z-10">
          <div className="p-3 bg-sky-500/10 rounded-xl">
            <TrendingUp className="w-5 h-5 text-sky-400" />
          </div>
          <div>
            <h4 className="text-sm font-bold text-white uppercase tracking-widest mb-2">Automated Optimization Insight</h4>
            <p className="text-sm text-slate-400 leading-relaxed max-w-2xl">
              By applying <span className="text-emerald-400 font-bold">{result.dosage} kg/acre</span> of <span className="text-emerald-400 font-bold">{result.recommended_fertilizer}</span> in approximately 
              <span className="text-white font-bold"> {result.apply_after_days} days</span>, the system predicts a refined yield output of 
              <span className="text-sky-400 font-bold"> {result.expected_yield_after_dosage} quintals per hectare</span>, accounting for registered <span className={result.pest_level === 'High' ? 'text-rose-400' : 'text-amber-400'}>{result.pest_level} pest risk</span>.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

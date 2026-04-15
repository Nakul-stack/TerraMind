import React from 'react';
import GraphRAGChat from '../components/GraphRAGChat';
import { Network, Database, BrainCircuit } from 'lucide-react';

function GraphRAG() {
  return (
    <div className="animate-in fade-in slide-in-from-bottom-4 duration-700">
      {/* Page Header */}
      <div className="mb-10">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
              <Network className="w-8 h-8 text-white" />
            </div>
            <div>
              <h2 className="text-3xl font-extrabold text-white tracking-tight">AugNosis Advisor</h2>
              <p className="text-sm text-slate-400 mt-1 max-w-xl leading-relaxed">
                Knowledge-graph powered crop advisory with unified LLM generation. 
                Deep multi-hop causal reasoning for complex agronomic issues.
              </p>
            </div>
          </div>
          
          <div className="flex gap-2">
            {[
              { icon: Database, label: 'Expert Verified', color: 'indigo' },
              { icon: BrainCircuit, label: 'LLM Inference', color: 'purple' }
            ].map((feature) => (
              <div key={feature.label} className={`flex items-center gap-2 px-4 py-2 rounded-xl bg-${feature.color}-500/5 border border-${feature.color}-500/10`}>
                <feature.icon className={`w-3.5 h-3.5 text-${feature.color}-400`} />
                <span className="text-[10px] font-bold text-white uppercase tracking-widest">{feature.label}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Info Banner */}
        <div className="mt-8 glass-card border-slate-800/50 bg-slate-800/20 p-4 border-l-4 border-l-indigo-500/50">
          <div className="flex items-start gap-3">
            <BrainCircuit className="w-5 h-5 text-indigo-400 shrink-0 mt-0.5" />
            <p className="text-xs text-slate-300 leading-relaxed">
               This module performs <span className="text-indigo-400 font-bold">multi-hop reasoning</span> across the localized agronomy knowledge graph. It synthesizes insights securely using the <span className="text-purple-400 font-bold">Qwen2.5</span> local language model. 
            </p>
          </div>
        </div>
      </div>
      
      <div className="max-w-6xl mx-auto space-y-8">
        <GraphRAGChat />
      </div>
    </div>
  );
}

export default GraphRAG;

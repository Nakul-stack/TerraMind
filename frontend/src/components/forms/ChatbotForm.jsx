import React, { useState } from 'react';
import { Send, ChevronDown, ChevronUp, MessageSquare, Info } from 'lucide-react';

export default function ChatbotForm({ onSubmit, isLoading, diagnosisCrop, diagnosisClass }) {
  const [question, setQuestion] = useState('');
  const [showContext, setShowContext] = useState(false);
  const [crop, setCrop] = useState(diagnosisCrop || '');
  const [diseaseClass, setDiseaseClass] = useState(diagnosisClass || '');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!question.trim() || isLoading) return;
    onSubmit({
      question: question.trim(),
      identified_crop: crop || null,
      identified_class: diseaseClass || null,
    });
    setQuestion('');
  };

  const formatDisease = (cls) => {
    if (!cls) return '';
    const parts = cls.split('__');
    const d = parts.length > 1 ? parts[1] : parts[0];
    return d.replace(/_/g, ' ');
  };

  return (
    <form onSubmit={handleSubmit} className="border-t border-slate-800/80 bg-slate-900/40 backdrop-blur-xl p-6">
      {/* Optional diagnosis context toggle */}
      {!diagnosisCrop && (
        <div className="mb-4">
          <button
            type="button"
            onClick={() => setShowContext(!showContext)}
            className="text-[10px] font-bold text-sky-400 hover:text-sky-300 flex items-center gap-1 uppercase tracking-widest transition-colors mb-2"
          >
            {showContext ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            Inquiry Context Overrides
          </button>

          {showContext && (
            <div className="mt-3 grid grid-cols-2 gap-4 animate-in fade-in slide-in-from-top-2 duration-300">
              <div className="flex flex-col gap-1.5">
                <input
                  type="text"
                  placeholder="Crop (e.g. Tomato)"
                  value={crop}
                  onChange={(e) => setCrop(e.target.value)}
                  className="bg-slate-950/50 border border-slate-800 rounded-xl px-4 py-2 text-xs text-white focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 outline-none transition-all placeholder:text-slate-700"
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <input
                  type="text"
                  placeholder="Disease ID (Tomato__Late_blight)"
                  value={diseaseClass}
                  onChange={(e) => setDiseaseClass(e.target.value)}
                  className="bg-slate-950/50 border border-slate-800 rounded-xl px-4 py-2 text-xs text-white focus:ring-2 focus:ring-sky-500/20 focus:border-sky-500 outline-none transition-all placeholder:text-slate-700"
                />
              </div>
            </div>
          )}
        </div>
      )}

      {/* Input + send */}
      <div className="flex gap-4 items-center">
        <div className="relative flex-1 group">
           <div className="absolute left-4 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-sky-500 transition-colors">
              <MessageSquare className="w-4 h-4" />
           </div>
          <input
            id="chatbot-input"
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={
              diagnosisCrop
                ? `Inquire about ${formatDisease(diagnosisClass)} protocols…`
                : 'Enter query for knowledge graph search…'
            }
            disabled={isLoading}
            className="w-full bg-slate-950/80 border border-slate-700/50 rounded-2xl pl-11 pr-4 py-3 text-sm text-white focus:ring-2 focus:ring-sky-500/30 focus:border-sky-500 outline-none disabled:bg-slate-900/50 disabled:cursor-not-allowed transition-all placeholder:text-slate-600 shadow-inner"
          />
        </div>
        
        <button
          id="chatbot-send-btn"
          type="submit"
          disabled={!question.trim() || isLoading}
          className="bg-sky-600 hover:bg-sky-500 disabled:bg-slate-800 disabled:text-slate-600 text-white p-3.5 rounded-2xl transition-all duration-300 shadow-lg shadow-sky-600/20 disabled:shadow-none flex items-center gap-2 group"
        >
          <Send className={`w-4 h-4 transition-transform ${question.trim() && !isLoading ? 'group-hover:translate-x-0.5 group-hover:-translate-y-0.5' : ''}`} />
          <span className="hidden sm:inline text-xs font-bold uppercase tracking-widest">Transmit</span>
        </button>
      </div>
      
      <div className="mt-3 flex items-center justify-center gap-2 opacity-50">
        <Info className="w-3 h-3 text-slate-500" />
        <p className="text-[9px] font-medium text-slate-500 uppercase tracking-tighter">
            AI responses are synthesized from curated agronomic pdf documentation.
        </p>
      </div>
    </form>
  );
}

import React, { useState, useRef, useEffect } from 'react';
import { useSearchParams, Link } from 'react-router-dom';
import { MessageSquare, ArrowLeft, Loader2, AlertCircle, RefreshCw, ShieldCheck, Lock, Sparkles } from 'lucide-react';
import { chatbotService } from '../services/chatbotService';
import ChatbotForm from '../components/forms/ChatbotForm';
import ChatMessageCard from '../components/results/ChatMessageCard';

export default function Chatbot() {
  const [searchParams] = useSearchParams();
  const crop = searchParams.get('crop');
  const disease = searchParams.get('disease');
  const report_id = searchParams.get('report_id');
  const hasContext = crop && disease;

  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const formatDisease = (cls) => {
    if (!cls) return '';
    const parts = cls.split('__');
    const d = parts.length > 1 ? parts[1] : parts[0];
    return d.replace(/_/g, ' ');
  };

  const handleSubmit = async ({ question, identified_crop, identified_class }) => {
    setError(null);
    const finalCrop = identified_crop || crop || null;
    const finalClass = identified_class || disease || null;

    const userMsg = { role: 'user', text: question };
    setMessages((prev) => [...prev, userMsg]);
    setIsLoading(true);

    try {
      const data = await chatbotService.askQuestion(question, 5, finalCrop, finalClass, report_id);

      const assistantMsg = {
        role: 'assistant',
        text: data.answer,
        allowed: data.allowed,
        reason: data.reason,
        sources: data.sources || [],
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err) {
      setError(err.message || 'Failed to get a response. Is the backend running?');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([]);
    setError(null);
  };

  // ── Gated Workflow: Require Diagnosis Report ─────────────────────────────
  if (!report_id) {
    return (
      <div className="max-w-xl mx-auto flex flex-col pt-24 items-center text-center animate-in fade-in slide-in-from-bottom-8 duration-1000">
        <div className="relative group mb-8">
            <div className="absolute inset-0 bg-rose-500/20 blur-3xl rounded-full group-hover:bg-rose-500/30 transition-all duration-700" />
            <div className="relative w-24 h-24 bg-slate-900 border border-slate-800 flex items-center justify-center rounded-3xl shadow-2xl">
                <Lock className="w-10 h-10 text-rose-500 animate-pulse" />
            </div>
        </div>
        
        <h2 className="text-3xl font-black text-white tracking-tight mb-4 uppercase italic">TerraBot Restricted</h2>
        <p className="text-slate-400 text-sm leading-relaxed mb-10 max-w-sm font-medium">
          The smart assistant requires specific diagnostic context to operate. 
          Please complete a <span className="text-emerald-400 font-bold underline underline-offset-4 decoration-emerald-500/30">disease diagnosis</span> and download the report to unlock this agent.
        </p>
        
        <Link
          to="/diagnosis"
          className="group relative px-8 py-3 bg-white hover:bg-slate-100 text-slate-950 text-xs font-black uppercase tracking-widest rounded-xl transition-all duration-300 shadow-xl shadow-white/10 flex items-center gap-3"
        >
          <ArrowLeft className="w-4 h-4 group-hover:-translate-x-1 transition-transform" />
          Go to Diagnosis
        </Link>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto flex flex-col h-[calc(100vh-140px)] overflow-y-auto custom-scrollbar animate-in fade-in duration-700">
      {/* Header Container */}
      <div className="mb-6 flex-shrink-0">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-4">
            <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-indigo-500 to-sky-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 relative group">
                <div className="absolute inset-0 bg-white/20 rounded-2xl scale-0 group-hover:scale-100 transition-transform duration-500" />
                <MessageSquare className="w-8 h-8 text-white relative z-10" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h2 className="text-2xl font-black text-white uppercase tracking-tight">TerraBot</h2>
                <div className="px-2 py-0.5 rounded-md bg-sky-500/10 border border-sky-500/20 text-[9px] font-black text-sky-400 uppercase tracking-tighter">Enterprise RAG</div>
              </div>
              <p className="text-xs text-slate-500 font-bold uppercase tracking-widest mt-1 italic">Knowledge-Graph Enhanced Advisory</p>
            </div>
          </div>
          
          <div className="flex items-center gap-3">
             <Link to="/diagnosis" className="flex items-center gap-2 px-4 py-2 rounded-xl bg-slate-800/50 hover:bg-slate-800 text-xs font-bold text-slate-300 border border-slate-700/50 transition-all duration-300">
                <ArrowLeft className="w-3.5 h-3.5" />
                Back to Results
             </Link>
             {messages.length > 0 && (
                <button
                  onClick={handleClearChat}
                  className="p-2.5 bg-slate-800/50 hover:bg-rose-500/10 text-slate-500 hover:text-rose-400 rounded-xl border border-slate-700/50 transition-all duration-300"
                  title="Clear chat"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
             )}
          </div>
        </div>
      </div>

      {/* Context Banner */}
      {hasContext && (
        <div className="mb-6 p-4 glass-card border-indigo-500/20 bg-indigo-500/5 relative overflow-hidden flex-shrink-0 group">
          <div className="absolute right-0 top-0 p-4 opacity-10 group-hover:scale-125 transition-transform duration-1000">
            <Sparkles className="w-12 h-12 text-indigo-400" />
          </div>
          <div className="flex items-center gap-4 relative z-10">
            <div className="p-2.5 bg-indigo-500/20 rounded-xl">
              <ShieldCheck className="w-5 h-5 text-indigo-400" />
            </div>
            <div>
                <p className="text-[10px] font-black text-indigo-400 uppercase tracking-[0.2em] mb-1">Active Diagnosis Session</p>
                <p className="text-sm font-bold text-white leading-none">
                    Analyzing <span className="text-sky-400">{crop}</span> — Pathology: <span className="text-rose-400">{formatDisease(disease)}</span>
                </p>
            </div>
          </div>
        </div>
      )}

      {/* Messages area */}
      <div className="flex-1 overflow-y-auto bg-slate-900/30 rounded-3xl border border-slate-800/50 p-6 space-y-6 min-h-0 custom-scrollbar mb-4 relative shadow-inner">
        {messages.length === 0 && !isLoading && (
          <div className="h-full flex flex-col items-center justify-center text-center animate-in zoom-in fade-in duration-1000">
            <div className="w-20 h-20 rounded-3xl bg-slate-800 flex items-center justify-center mb-6 shadow-2xl border border-slate-700 relative overflow-hidden group">
                <div className="absolute inset-0 bg-indigo-500/5 animate-pulse" />
                <MessageSquare className="w-10 h-10 text-slate-600 relative group-hover:scale-110 transition-transform duration-500" />
            </div>
            <h3 className="text-xl font-black text-white uppercase tracking-tight mb-2 italic">Awaiting Directives</h3>
            <p className="text-sm text-slate-500 max-w-sm leading-relaxed font-medium">
              Inquire about the recently synthesized expert report, request treatment deep-dives, or verify agronomic citations from the knowledge graph.
            </p>
          </div>
        )}

        {messages.map((msg, idx) => (
          <ChatMessageCard key={idx} message={msg} />
        ))}

        {/* Loading indicator */}
        {isLoading && (
          <div className="flex gap-4 justify-start animate-pulse">
            <div className="flex-shrink-0 w-10 h-10 rounded-2xl bg-slate-800 border border-slate-700 flex items-center justify-center mt-1">
              <Loader2 className="w-5 h-5 text-indigo-400 animate-spin" />
            </div>
            <div className="bg-slate-800/50 border border-slate-700/30 rounded-2xl rounded-tl-none px-5 py-3 flex items-center gap-3">
              <div className="flex gap-1.5">
                <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <div className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce" />
              </div>
              <span className="text-xs font-bold text-slate-500 uppercase tracking-widest">Traversing AgroKG...</span>
            </div>
          </div>
        )}

        {error && (
          <div className="flex items-start gap-4 p-4 glass-card border-rose-500/20 bg-rose-500/5 text-rose-400 animate-in shake duration-500">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <div className="text-xs font-bold uppercase tracking-tight">
              <p className="mb-1">Neural Service Disruption</p>
              <p className="opacity-70">{error}</p>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="flex-shrink-0 rounded-t-3xl overflow-hidden shadow-[0_-10px_30px_rgba(0,0,0,0.3)]">
        <ChatbotForm
          onSubmit={handleSubmit}
          isLoading={isLoading}
          diagnosisCrop={crop}
          diagnosisClass={disease}
        />
      </div>
    </div>
  );
}

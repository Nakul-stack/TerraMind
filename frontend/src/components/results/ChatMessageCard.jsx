import React from 'react';
import { User, Bot, AlertTriangle, ShieldCheck } from 'lucide-react';
import ReactMarkdown from 'react-markdown';

export default function ChatMessageCard({ message }) {
  const isUser = message.role === 'user';
  const isRefusal = message.role === 'assistant' && message.allowed === false;

  return (
    <div className={`flex gap-4 ${isUser ? 'justify-end' : 'justify-start'} mb-4 animate-in fade-in slide-in-from-bottom-2 duration-300`}>
      {/* Avatar — assistant only (left side) */}
      {!isUser && (
        <div className={`flex-shrink-0 w-10 h-10 rounded-2xl flex items-center justify-center mt-1 border transition-all duration-300 ${
          isRefusal 
          ? 'bg-amber-500/10 text-amber-500 border-amber-500/20' 
          : 'bg-indigo-500/10 text-indigo-400 border-indigo-500/20'
        }`}>
          {isRefusal ? <AlertTriangle className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
        </div>
      )}

      {/* Bubble */}
      <div className={`max-w-[85%] flex flex-col ${isUser ? 'items-end' : 'items-start'}`}>
        <div
          className={`rounded-2xl px-5 py-3 text-sm leading-relaxed relative ${
            isUser
              ? 'bg-gradient-to-br from-indigo-600 to-indigo-700 text-white rounded-tr-none shadow-lg shadow-indigo-600/20 border border-indigo-500/20'
              : isRefusal
                ? 'bg-amber-500/5 text-amber-200 border border-amber-500/20 rounded-tl-none pr-6'
                : 'bg-slate-800/80 text-slate-200 border border-slate-700/50 rounded-tl-none backdrop-blur-md pr-6'
          }`}
        >
           {/* Decorative corner for assistant */}
           {!isUser && (
             <div className="absolute -left-[1px] -top-[1px] w-4 h-4 rounded-tl-2xl border-l border-t border-inherit" />
           )}

          {message.text ? (
            isUser ? (
              <p className="font-medium">{message.text}</p>
            ) : (
              <div className="prose prose-invert prose-sm max-w-none prose-p:leading-relaxed prose-strong:text-white prose-strong:font-bold prose-code:text-sky-300 prose-code:bg-slate-900 prose-code:px-1 prose-code:rounded prose-code:font-mono">
                <ReactMarkdown
                  components={{
                    strong: ({ node, ...props }) => <strong className="font-bold text-white bg-white/5 px-1 rounded" {...props} />
                  }}
                >
                  {message.text}
                </ReactMarkdown>
              </div>
            )
          ) : null}
        </div>

        {/* Status flags & labels */}
        <div className="flex items-center gap-2 mt-2 px-1">
          {isRefusal && message.reason && message.reason !== 'ok' && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 bg-amber-500/10 text-amber-500 text-[10px] rounded-full font-bold uppercase tracking-wider border border-amber-500/20">
              <AlertTriangle className="w-2.5 h-2.5" />
              {message.reason.replace(/_/g, ' ')}
            </span>
          )}
          {!isUser && !isRefusal && (
            <span className="flex items-center gap-1.5 px-2 py-0.5 bg-emerald-500/10 text-emerald-400 text-[9px] rounded-full font-bold uppercase tracking-widest border border-emerald-500/10">
                <ShieldCheck className="w-2.5 h-2.5" />
                Verified Context
            </span>
          )}
          <span className="text-[9px] text-slate-600 font-bold uppercase tracking-tighter">
             {isUser ? 'Agronomist' : 'Smart Assistant'}
          </span>
        </div>
      </div>

      {/* Avatar — user only (right side) */}
      {isUser && (
        <div className="flex-shrink-0 w-10 h-10 rounded-2xl bg-slate-800 border border-slate-700 text-slate-400 flex items-center justify-center mt-1 shadow-lg">
          <User className="w-5 h-5" />
        </div>
      )}
    </div>
  );
}

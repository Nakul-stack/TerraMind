import React, { useState } from 'react';
import { FileText, ChevronDown, ChevronUp } from 'lucide-react';

export default function SourceCitationCard({ source }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="bg-white border border-gray-200 rounded-lg px-3 py-2 text-xs shadow-sm">
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-1.5 min-w-0">
          <FileText className="w-3.5 h-3.5 text-indigo-500 flex-shrink-0" />
          <span className="font-medium text-gray-700 truncate" title={source.file_name}>
            {source.file_name}
          </span>
          {source.page && (
            <span className="text-gray-400 flex-shrink-0">p.{source.page}</span>
          )}
          {source.score > 0 && (
            <span className="text-indigo-500 flex-shrink-0 font-medium">
              {(source.score * 100).toFixed(0)}%
            </span>
          )}
        </div>
        <button
          type="button"
          onClick={() => setExpanded(!expanded)}
          className="text-gray-400 hover:text-gray-600 flex-shrink-0"
          aria-label={expanded ? 'Collapse snippet' : 'Expand snippet'}
        >
          {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>
      </div>

      {expanded && source.snippet && (
        <p className="mt-1.5 text-gray-600 leading-relaxed border-t border-gray-100 pt-1.5">
          {source.snippet}
        </p>
      )}
    </div>
  );
}

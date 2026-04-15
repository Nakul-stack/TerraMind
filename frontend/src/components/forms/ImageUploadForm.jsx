import React, { useState, useRef } from 'react';
import { Upload, Image as ImageIcon, X, Zap, Loader2, Search } from 'lucide-react';
import { DIAGNOSIS_DEFAULT_TOP_K, DIAGNOSIS_TOPK_OPTIONS } from '../../config/runtimeConfig';

export default function ImageUploadForm({ onSubmit, isLoading }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [topK, setTopK] = useState(DIAGNOSIS_DEFAULT_TOP_K);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  const handleFileChange = (file) => {
    if (!file) return;
    const allowed = ['image/jpeg', 'image/png', 'image/webp', 'image/jpg'];
    if (!allowed.includes(file.type)) {
      alert('Please upload a JPEG, PNG, or WebP image.');
      return;
    }
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleInputChange = (e) => {
    handleFileChange(e.target.files[0]);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileChange(e.dataTransfer.files[0]);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!selectedFile) return;
    onSubmit(selectedFile, topK);
  };

  return (
    <div className="glass-card overflow-hidden border-emerald-500/10">
      <div className="bg-gradient-to-r from-emerald-500/10 via-transparent to-transparent p-6 border-b border-slate-800/50 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-extrabold text-white flex items-center gap-2">
            <ImageIcon className="w-5 h-5 text-emerald-500" />
            Visual Symptom Capture
          </h3>
          <p className="text-[10px] text-slate-500 mt-1 uppercase tracking-widest font-bold">Upload leaf/plant imagery for neural diagnosis</p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20">
          <Zap className="w-3 h-3 text-emerald-400" />
          <span className="text-[9px] font-bold text-emerald-400 uppercase tracking-widest">GPU Accelerated</span>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="p-6">
        {/* Drop zone */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            relative border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all duration-300 group
            ${dragActive
              ? 'border-emerald-500 bg-emerald-500/10'
              : 'border-slate-800 hover:border-emerald-500/50 hover:bg-slate-800/40'
            }
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png,image/webp"
            onChange={handleInputChange}
            className="hidden"
            id="diagnosis-image-upload"
          />

          {previewUrl ? (
            <div className="relative inline-block animate-in fade-in zoom-in duration-300">
              <div className="relative">
                <div className="absolute inset-0 bg-emerald-500/20 blur-xl opacity-0 group-hover:opacity-100 transition-opacity" />
                <img
                  src={previewUrl}
                  alt="Selected preview"
                  className="max-h-64 rounded-xl mx-auto shadow-2xl border border-slate-700/50 relative z-10"
                />
              </div>
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); clearFile(); }}
                className="absolute -top-3 -right-3 bg-rose-500 hover:bg-rose-600 text-white rounded-full p-2 shadow-xl shadow-rose-500/20 transition-all hover:scale-110 z-20"
              >
                <X className="w-4 h-4" />
              </button>
              <div className="mt-4 flex items-center justify-center gap-2">
                <span className="px-3 py-1 bg-slate-800 border border-slate-700 rounded-full text-xs text-slate-400 font-medium">
                  {selectedFile?.name}
                </span>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center gap-4 py-6">
              <div className="w-16 h-16 rounded-3xl bg-emerald-500/10 flex items-center justify-center border border-emerald-500/10 group-hover:scale-110 transition-transform duration-500">
                <Upload className="w-8 h-8 text-emerald-500" />
              </div>
              <div className="space-y-1">
                <p className="text-base font-bold text-white">Capture or Drop Imagery</p>
                <p className="text-sm text-slate-400">Click to browse your local directory</p>
              </div>
              <div className="flex gap-2 mt-2">
                {['JPEG', 'PNG', 'WEBP'].map(type => (
                  <span key={type} className="px-2 py-0.5 rounded bg-slate-800 border border-slate-700 text-[10px] text-slate-500 font-bold">{type}</span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Options & Submit */}
        <div className="mt-10 pt-8 border-t border-slate-800/50 flex flex-col sm:flex-row items-center justify-between gap-6">
          <div className="flex flex-col gap-2 w-full sm:w-auto">
            <label htmlFor="top-k-select" className="text-[10px] font-bold text-slate-500 uppercase tracking-widest px-1 text-center sm:text-left">
              Inference Depth
            </label>
            <div className="flex bg-slate-900/50 p-1 rounded-xl border border-slate-800/50">
              {DIAGNOSIS_TOPK_OPTIONS.map((k) => (
                <button
                  key={k}
                  type="button"
                  onClick={() => setTopK(k)}
                  className={`px-4 py-1.5 rounded-lg text-xs font-bold transition-all ${
                    topK === k 
                    ? 'bg-emerald-500 text-white shadow-lg shadow-emerald-500/20' 
                    : 'text-slate-500 hover:text-slate-300'
                  }`}
                >
                  Top {k}
                </button>
              ))}
            </div>
          </div>

          <button
            type="submit"
            disabled={isLoading || !selectedFile}
            className="w-full sm:w-auto group relative px-10 py-3 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-bold rounded-xl transition-all duration-300 disabled:opacity-50 overflow-hidden shadow-lg shadow-emerald-500/20"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-400/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
            <span className="relative flex items-center justify-center gap-2">
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Running Neural Analysis...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  Initiate Diagnosis
                </>
              )}
            </span>
          </button>
        </div>
      </form>
    </div>
  );
}

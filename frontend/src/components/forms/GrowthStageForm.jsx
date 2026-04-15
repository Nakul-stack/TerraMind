import React, { useState } from 'react';
import { Thermometer, Droplets, Wind, Beaker, TrendingUp, CloudRain, Sprout, LayoutGrid, Zap, Activity } from 'lucide-react';

const InputField = ({ label, name, icon: Icon, type = 'number', value, onChange, step = "0.1", required = true }) => (
  <div className="flex flex-col gap-1.5 group">
    <label htmlFor={name} className="text-[10px] font-bold text-slate-500 uppercase tracking-widest px-1">
      {label}
    </label>
    <div className="relative">
      <div className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-emerald-500 transition-colors">
        <Icon className="w-4 h-4" />
      </div>
      <input
        type={type}
        id={name}
        name={name}
        step={step}
        value={value}
        onChange={onChange}
        required={required}
        placeholder="0.0"
        className="w-full bg-slate-900/50 border border-slate-700/50 rounded-xl py-2.5 pl-10 pr-4 text-sm text-white focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 outline-none transition-all placeholder:text-slate-700"
      />
    </div>
  </div>
);

const SelectField = ({ label, name, icon: Icon, value, onChange, options, required = true }) => (
  <div className="flex flex-col gap-1.5 group">
    <label htmlFor={name} className="text-[10px] font-bold text-slate-500 uppercase tracking-widest px-1">
      {label}
    </label>
    <div className="relative">
      <div className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 group-focus-within:text-emerald-500 transition-colors">
        <Icon className="w-4 h-4" />
      </div>
      <select
        id={name}
        name={name}
        value={value}
        onChange={onChange}
        required={required}
        className="w-full bg-slate-900/50 border border-slate-700/50 rounded-xl py-2.5 pl-10 pr-4 text-sm text-white focus:ring-2 focus:ring-emerald-500/20 focus:border-emerald-500 outline-none transition-all appearance-none cursor-pointer"
      >
        <option value="" disabled className="bg-slate-900">Select {label}</option>
        {options.map(opt => (
          <option key={opt.value} value={opt.value} className="bg-slate-900">{opt.label}</option>
        ))}
      </select>
      <div className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-600 pointer-events-none">
        <TrendingUp className="w-3 h-3 rotate-90" />
      </div>
    </div>
  </div>
);

export default function GrowthStageForm({ onSubmit, isLoading }) {
  const [formData, setFormData] = useState({
    temperature: '', humidity: '', moisture: '',
    soil_type: '', crop_type: '',
    N: '', P: '', K: '', ph: '', rainfall: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const parsedData = {
      ...formData,
      temperature: parseFloat(formData.temperature),
      humidity: parseFloat(formData.humidity),
      moisture: parseFloat(formData.moisture),
      N: parseFloat(formData.N),
      P: parseFloat(formData.P),
      K: parseFloat(formData.K),
      ph: parseFloat(formData.ph),
      rainfall: parseFloat(formData.rainfall),
    };

    onSubmit(parsedData);
  };

  return (
    <div className="glass-card overflow-hidden border-emerald-500/10">
      <div className="bg-gradient-to-r from-emerald-500/10 via-transparent to-transparent p-6 border-b border-slate-800/50">
        <h3 className="text-lg font-extrabold text-white flex items-center gap-2">
          <Activity className="w-5 h-5 text-emerald-500" />
          Real-time Crop Metrics
        </h3>
        <p className="text-xs text-slate-500 mt-1 uppercase tracking-widest font-bold">Input current field conditions for causal analysis</p>
      </div>

      <form onSubmit={handleSubmit} className="p-6">
        <div className="space-y-8">
          {/* Section 1: Environmental */}
          <div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
              <InputField label="Temp (°C)" name="temperature" icon={Thermometer} value={formData.temperature} onChange={handleChange} />
              <InputField label="Humidity (%)" name="humidity" icon={Wind} value={formData.humidity} onChange={handleChange} />
              <InputField label="Soil Moisture (%)" name="moisture" icon={Droplets} value={formData.moisture} onChange={handleChange} />
            </div>
          </div>

          {/* Section 2: Soil Nutrients */}
          <div className="pt-4 border-t border-slate-800/30">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
              <InputField label="Nitrogen (N) (kg/ha)" name="N" icon={Zap} value={formData.N} onChange={handleChange} />
              <InputField label="Phosphorus (P) (kg/ha)" name="P" icon={Zap} value={formData.P} onChange={handleChange} />
              <InputField label="Potassium (K) (kg/ha)" name="K" icon={Zap} value={formData.K} onChange={handleChange} />
              <InputField label="pH Level" name="ph" icon={Beaker} value={formData.ph} onChange={handleChange} />
            </div>
          </div>

          {/* Section 3: Contextual */}
          <div className="pt-4 border-t border-slate-800/30">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <InputField label="Rainfall (mm)" name="rainfall" icon={CloudRain} value={formData.rainfall} onChange={handleChange} />
              <SelectField
                label="Soil Type" name="soil_type" icon={LayoutGrid} value={formData.soil_type} onChange={handleChange}
                options={[
                  {value: 'Loamy', label: 'Loamy'}, {value: 'Sandy', label: 'Sandy'},
                  {value: 'Clay', label: 'Clay'}, {value: 'Black', label: 'Black'}, {value: 'Red', label: 'Red'}
                ]}
              />
              <SelectField
                label="Crop Type" name="crop_type" icon={Sprout} value={formData.crop_type} onChange={handleChange}
                options={[
                  {value: 'Sugarcane', label: 'Sugarcane'}, {value: 'Cotton', label: 'Cotton'},
                  {value: 'Millets', label: 'Millets'}, {value: 'Paddy', label: 'Paddy'},
                  {value: 'Pulses', label: 'Pulses'}, {value: 'Wheat', label: 'Wheat'},
                  {value: 'Tobacco', label: 'Tobacco'}, {value: 'Barley', label: 'Barley'},
                  {value: 'Oil seeds', label: 'Oil seeds'}, {value: 'Ground Nuts', label: 'Ground Nuts'},
                  {value: 'Maize', label: 'Maize'}
                ]}
              />
            </div>
          </div>
        </div>

        <div className="mt-10 flex justify-end">
          <button
            type="submit"
            disabled={isLoading}
            className="group relative px-8 py-3 bg-emerald-600 hover:bg-emerald-500 text-white text-sm font-bold rounded-xl transition-all duration-300 disabled:opacity-50 overflow-hidden shadow-lg shadow-emerald-500/20"
          >
            <div className="absolute inset-0 bg-gradient-to-r from-emerald-400/20 to-transparent translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-700" />
            <span className="relative flex items-center gap-2">
              {isLoading ? (
                <>
                  <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                  Analyzing Pipeline...
                </>
              ) : (
                <>
                  <Activity className="w-4 h-4" />
                  Generate Stage Advisory
                </>
              )}
            </span>
          </button>
        </div>
      </form>
    </div>
  );
}

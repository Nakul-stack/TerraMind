import React, { useState } from 'react';
import stateDistrictData from '../../data/state_district_mapping.json';

const InputField = ({ label, name, type = 'number', value, onChange, step = "0.1", required = true, placeholder = "" }) => (
  <div className="flex flex-col gap-1.5">
    <label htmlFor={name} className="text-sm font-semibold text-gray-600 tracking-wide uppercase">{label}</label>
    <input
      type={type}
      id={name}
      name={name}
      step={step}
      value={value}
      onChange={onChange}
      required={required}
      placeholder={placeholder}
      className="border border-gray-200 rounded-lg px-3 py-2.5 bg-gray-50 focus:bg-white focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 outline-none transition-all text-sm"
    />
  </div>
);

const SelectField = ({ label, name, value, onChange, options, required = true, disabled = false }) => (
  <div className="flex flex-col gap-1.5">
    <label htmlFor={name} className="text-sm font-semibold text-gray-600 tracking-wide uppercase">{label}</label>
    <select
      id={name}
      name={name}
      value={value}
      onChange={onChange}
      required={required}
      disabled={disabled}
      className="border border-gray-200 rounded-lg px-3 py-2.5 bg-gray-50 focus:bg-white focus:ring-2 focus:ring-emerald-500 outline-none transition-all text-sm disabled:bg-gray-100 disabled:text-gray-400 disabled:cursor-not-allowed"
    >
      <option value="" disabled>Select {label}</option>
      {options.map(opt => (
        <option key={opt.value} value={opt.value}>{opt.label}</option>
      ))}
    </select>
  </div>
);

export default function BeforeSowingForm({ onSubmit, isLoading }) {
  const [formData, setFormData] = useState({
    N: '', P: '', K: '',
    temperature: '', humidity: '', rainfall: '', ph: '',
    soil_type: '', season: '', state_name: '', district_name: '', area: ''
  });

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleStateChange = (e) => {
    const state_name = e.target.value;
    setFormData(prev => ({ ...prev, state_name, district_name: '' }));
  };

  const stateOptions = Object.keys(stateDistrictData).sort().map(s => ({ value: s, label: s }));
  const districtOptions = formData.state_name
    ? (stateDistrictData[formData.state_name] || []).map(d => ({ value: d, label: d }))
    : [];

  const handleSubmit = (e) => {
    e.preventDefault();
    const parsedData = {
      ...formData,
      N: parseFloat(formData.N),
      P: parseFloat(formData.P),
      K: parseFloat(formData.K),
      temperature: parseFloat(formData.temperature),
      humidity: parseFloat(formData.humidity),
      rainfall: parseFloat(formData.rainfall),
      ph: parseFloat(formData.ph),
      area: formData.area ? parseFloat(formData.area) : undefined,
    };
    onSubmit(parsedData);
  };

  return (
    <form onSubmit={handleSubmit} className="bg-white rounded-2xl shadow-sm border border-gray-100 overflow-hidden">
      {/* Soil & NPK Section */}
      <div className="p-6 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-5">
          <div className="w-2 h-2 rounded-full bg-emerald-500"></div>
          <h3 className="text-base font-bold text-gray-800">Soil & Weather Parameters</h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          <InputField label="Nitrogen (N) (kg/ha)" name="N" value={formData.N} onChange={handleChange} placeholder="e.g. 90 kg/ha" />
          <InputField label="Phosphorus (P) (kg/ha)" name="P" value={formData.P} onChange={handleChange} placeholder="e.g. 42 kg/ha" />
          <InputField label="Potassium (K) (kg/ha)" name="K" value={formData.K} onChange={handleChange} placeholder="e.g. 43 kg/ha" />
          <InputField label="pH Level" name="ph" value={formData.ph} onChange={handleChange} placeholder="e.g. 6.5" />
          <InputField label="Temperature (°C)" name="temperature" value={formData.temperature} onChange={handleChange} placeholder="e.g. 25" />
          <InputField label="Humidity (%)" name="humidity" value={formData.humidity} onChange={handleChange} placeholder="e.g. 80" />
          <InputField label="Rainfall (mm)" name="rainfall" value={formData.rainfall} onChange={handleChange} placeholder="e.g. 200" />
          <InputField label="Area (ha)" name="area" value={formData.area} onChange={handleChange} required={false} placeholder="optional" />
        </div>
      </div>

      {/* Location Section */}
      <div className="p-6 border-b border-gray-100">
        <div className="flex items-center gap-2 mb-5">
          <div className="w-2 h-2 rounded-full bg-blue-500"></div>
          <h3 className="text-base font-bold text-gray-800">Location & Season</h3>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <SelectField
            label="Soil Type" name="soil_type" value={formData.soil_type} onChange={handleChange}
            options={[
              {value: 'loamy', label: 'Loamy'}, {value: 'sandy', label: 'Sandy'},
              {value: 'clay', label: 'Clay'}, {value: 'silt', label: 'Silt'},
              {value: 'black', label: 'Black'}, {value: 'red', label: 'Red'},
            ]}
          />
          <SelectField
            label="Season" name="season" value={formData.season} onChange={handleChange}
            options={[
              {value: 'kharif', label: 'Kharif'}, {value: 'rabi', label: 'Rabi'},
              {value: 'zaid', label: 'Zaid'}, {value: 'annual', label: 'Annual / Whole Year'},
            ]}
          />
          <SelectField
            label="State" name="state_name" value={formData.state_name}
            onChange={handleStateChange} options={stateOptions}
          />
          <SelectField
            label="District" name="district_name" value={formData.district_name}
            onChange={handleChange} options={districtOptions}
            disabled={!formData.state_name}
          />
        </div>
      </div>

      {/* Submit */}
      <div className="p-6 bg-gray-50/50 flex justify-end">
        <button
          type="submit"
          disabled={isLoading}
          className="bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-700 hover:to-teal-700 text-white font-semibold py-3 px-8 rounded-xl shadow-lg shadow-emerald-200 transition-all disabled:opacity-50 disabled:shadow-none flex items-center gap-2"
        >
          {isLoading ? (
            <>
              <svg className="animate-spin h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Running Pipeline…
            </>
          ) : (
            <>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
              Generate Advisory
            </>
          )}
        </button>
      </div>
    </form>
  );
}

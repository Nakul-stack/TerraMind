import React from 'react';
import { Leaf, Droplets, Sun, Activity, Sprout, TrendingUp, MapPin, BarChart3, Info } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  PieChart, Pie, Legend
} from 'recharts';

const COLORS = ['#059669', '#0891b2', '#7c3aed', '#d97706', '#dc2626'];
const CONFIDENCE_COLORS = ['#059669', '#10b981', '#6ee7b7'];

// ---------------------------------------------------------------------------
// Sub-Components
// ---------------------------------------------------------------------------

const SectionHeader = ({ icon: Icon, title, color = 'emerald' }) => (
  <div className="flex items-center gap-2 mb-4">
    <div className={`p-1.5 rounded-lg bg-${color}-50`}>
      <Icon className={`w-5 h-5 text-${color}-600`} />
    </div>
    <h3 className="text-lg font-bold text-gray-800">{title}</h3>
  </div>
);

const StatCard = ({ label, value, sub, color = 'gray' }) => (
  <div className={`p-4 rounded-xl bg-${color}-50 border border-${color}-100`}>
    <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-1">{label}</p>
    <p className={`text-xl font-bold text-${color}-700`}>{value}</p>
    {sub && <p className="text-xs text-gray-500 mt-1">{sub}</p>}
  </div>
);

const ConfidenceBar = ({ label, value, maxValue = 1.0 }) => {
  const pct = Math.min(100, (value / maxValue) * 100);
  return (
    <div className="flex items-center gap-3">
      <span className="text-sm font-medium text-gray-700 w-28 truncate capitalize">{label}</span>
      <div className="flex-1 h-3 bg-gray-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, #059669, #10b981)`,
          }}
        />
      </div>
      <span className="text-sm font-bold text-gray-700 w-14 text-right">{(value * 100).toFixed(1)}%</span>
    </div>
  );
};

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function BeforeSowingResultCard({ result }) {
  if (!result || !result.success) return null;

  const crop = result.crop_recommender || {};
  const yld = result.yield_predictor || {};
  const agri = result.agri_condition_advisor || {};
  const di = result.district_intelligence || {};
  const notes = result.system_notes || [];

  // Chart data
  const cropChartData = (crop.top_3 || []).map((c, i) => ({
    name: c.crop,
    confidence: parseFloat((c.confidence * 100).toFixed(1)),
    fill: CONFIDENCE_COLORS[i] || CONFIDENCE_COLORS[0],
  }));

  const competingCrops = (di.top_competing_crops || []).slice(0, 5).map((c, i) => ({
    name: c, value: 5 - i, fill: COLORS[i],
  }));

  return (
    <div className="mt-8 space-y-6 animate-in fade-in duration-500">

      {/* ── Crop Recommendation ─────────────────────────────────────── */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <SectionHeader icon={Leaf} title="Crop Recommendation" color="emerald" />

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Selected crop hero card */}
          <div className="bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl p-6 text-white">
            <p className="text-sm font-medium opacity-80 uppercase tracking-wider">Best Crop Match</p>
            <p className="text-3xl font-bold mt-1 capitalize">{crop.selected_crop || 'N/A'}</p>
            <p className="text-lg mt-2 opacity-90">
              {crop.selected_confidence ? `${(crop.selected_confidence * 100).toFixed(1)}% confidence` : ''}
            </p>
          </div>

          {/* Top-3 confidence bars */}
          <div className="space-y-3 flex flex-col justify-center">
            <p className="text-sm font-semibold text-gray-500 uppercase tracking-wide">Top 3 Predictions</p>
            {(crop.top_3 || []).map((c, i) => (
              <ConfidenceBar key={i} label={c.crop} value={c.confidence} />
            ))}
          </div>
        </div>

        {/* Bar chart */}
        {cropChartData.length > 0 && (
          <div className="mt-6 h-48">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={cropChartData} layout="vertical">
                <XAxis type="number" domain={[0, 100]} tick={{ fontSize: 12 }} unit="%" />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 13 }} width={100} />
                <Tooltip formatter={(v) => `${v}%`} />
                <Bar dataKey="confidence" radius={[0, 6, 6, 0]}>
                  {cropChartData.map((entry, i) => (
                    <Cell key={i} fill={entry.fill} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}
      </div>

      {/* ── Yield Prediction ────────────────────────────────────────── */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <SectionHeader icon={TrendingUp} title="Yield Prediction" color="blue" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <StatCard
            label="Expected Yield"
            value={`${yld.expected_yield?.toFixed(2) || '0.00'} ${yld.unit || 't/ha'}`}
            color="blue"
          />
          <StatCard
            label="Lower Bound"
            value={`${yld.confidence_band?.lower?.toFixed(2) || '0.00'} ${yld.unit || 't/ha'}`}
            sub="Conservative estimate"
            color="gray"
          />
          <StatCard
            label="Upper Bound"
            value={`${yld.confidence_band?.upper?.toFixed(2) || '0.00'} ${yld.unit || 't/ha'}`}
            sub="Optimistic estimate"
            color="gray"
          />
        </div>
        {yld.explanation && (
          <div className="mt-4 p-3 bg-blue-50 border border-blue-100 rounded-lg text-sm text-blue-800">
            <Info className="w-4 h-4 inline mr-1.5 -mt-0.5" />
            {yld.explanation}
          </div>
        )}
      </div>

      {/* ── Agri-Condition Advisory ─────────────────────────────────── */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <SectionHeader icon={Droplets} title="Pre-Sowing Condition Advisory" color="cyan" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl bg-amber-50 border border-amber-100 flex items-start gap-3">
            <Sun className="w-6 h-6 text-amber-500 mt-0.5" />
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase">Sunlight</p>
              <p className="text-xl font-bold text-amber-700">{agri.sunlight_hours ?? '—'} hrs/day</p>
            </div>
          </div>
          <div className="p-4 rounded-xl bg-sky-50 border border-sky-100 flex items-start gap-3">
            <Droplets className="w-6 h-6 text-sky-500 mt-0.5" />
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase">Irrigation Type</p>
              <p className="text-xl font-bold text-sky-700 capitalize">{agri.irrigation_type || '—'}</p>
            </div>
          </div>
          <div className="p-4 rounded-xl bg-violet-50 border border-violet-100 flex items-start gap-3">
            <Activity className="w-6 h-6 text-violet-500 mt-0.5" />
            <div>
              <p className="text-xs font-semibold text-gray-500 uppercase">Irrigation Need</p>
              <p className="text-xl font-bold text-violet-700 capitalize">{agri.irrigation_need || '—'}</p>
            </div>
          </div>
        </div>

        {agri.explanation && (
          <div className="mt-4 p-3 bg-cyan-50 border border-cyan-100 rounded-lg text-sm text-cyan-800">
            {agri.explanation}
          </div>
        )}

        {agri.district_prior_used && agri.irrigation_reasoning && (
          <div className="mt-3 p-3 bg-indigo-50 border border-indigo-100 rounded-lg text-sm text-indigo-800">
            <span className="font-semibold">District Prior: </span>
            {agri.irrigation_reasoning}
          </div>
        )}
      </div>

      {/* ── District Intelligence ───────────────────────────────────── */}
      <div className="bg-white rounded-2xl shadow-sm border border-gray-100 p-6">
        <SectionHeader icon={MapPin} title="District Intelligence" color="purple" />

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <StatCard
            label="Crop Area Share"
            value={di.district_crop_share_percent != null ? `${di.district_crop_share_percent}%` : 'N/A'}
            sub="of district cultivated area"
            color="purple"
          />
          <StatCard
            label="Yield Trend"
            value={di.yield_trend || 'N/A'}
            sub="Historical direction"
            color="emerald"
          />
          <StatCard
            label="Best Season"
            value={di.best_historical_season || 'N/A'}
            sub="Historically optimal"
            color="blue"
          />
          <StatCard
            label="Irrigated Area"
            value={di.crop_irrigated_area_percent != null ? `${di.crop_irrigated_area_percent}%` : 'N/A'}
            sub="of crop area irrigated"
            color="cyan"
          />
        </div>

        {/* Key Insights — human-readable output from ICRISAT engine */}
        {(di.insights || []).length > 0 && (
          <div className="mb-5 p-4 bg-purple-50 border border-purple-100 rounded-xl">
            <p className="text-sm font-bold text-purple-800 mb-3 flex items-center gap-1.5">
              <Info className="w-4 h-4" /> Key Insights from ICRISAT District Data
            </p>
            <ul className="space-y-2">
              {(di.insights || []).map((insight, i) => (
                <li key={i} className="text-sm text-purple-900 flex items-start gap-2">
                  <span className="text-purple-400 mt-0.5 shrink-0">▸</span>
                  {insight}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* 10-year trajectory */}
        {di.ten_year_trajectory_summary && (
          <div className="p-4 bg-gray-50 rounded-xl border border-gray-100 mb-4">
            <p className="text-sm font-semibold text-gray-600 mb-1 flex items-center gap-1.5">
              <BarChart3 className="w-4 h-4" /> 10-Year Yield Trajectory
            </p>
            <p className="text-sm text-gray-700">{di.ten_year_trajectory_summary}</p>
          </div>
        )}

        {/* Irrigation Infrastructure with Pie Chart */}
        {di.irrigation_infrastructure_summary && (
          <div className="p-4 bg-gray-50 rounded-xl border border-gray-100 mb-4">
            <p className="text-sm font-semibold text-gray-600 mb-2 flex items-center gap-1.5">
              <Droplets className="w-4 h-4" /> Irrigation Infrastructure
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <p className="text-sm text-gray-700">{di.irrigation_infrastructure_summary}</p>
              {di.irrigation_infrastructure_breakdown && Object.keys(di.irrigation_infrastructure_breakdown).length > 0 && (
                <div className="h-44">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={Object.entries(di.irrigation_infrastructure_breakdown).map(([name, pct], i) => ({
                          name, value: pct, fill: COLORS[i % COLORS.length],
                        }))}
                        cx="50%" cy="50%"
                        innerRadius={35}
                        outerRadius={65}
                        paddingAngle={3}
                        dataKey="value"
                        label={({name, value}) => `${name}: ${value}%`}
                      >
                        {Object.entries(di.irrigation_infrastructure_breakdown).map(([, ], i) => (
                          <Cell key={i} fill={COLORS[i % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip formatter={(v) => `${v}%`} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Competing Crops */}
        {competingCrops.length > 0 && (
          <div>
            <p className="text-sm font-semibold text-gray-600 mb-3 flex items-center gap-1.5">
              <Sprout className="w-4 h-4" /> Top Competing Crops in District
            </p>
            <div className="flex flex-wrap gap-2">
              {(di.top_competing_crops || []).slice(0, 8).map((c, i) => (
                <span key={i} className="px-3 py-1.5 bg-gray-100 text-gray-700 rounded-full text-sm font-medium capitalize">
                  {c}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* ── System Notes ────────────────────────────────────────────── */}
      {notes.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-xl p-4">
          <p className="text-sm font-semibold text-amber-800 mb-2">System Notes</p>
          <ul className="space-y-1">
            {notes.map((note, i) => (
              <li key={i} className="text-sm text-amber-700 flex items-start gap-2">
                <span className="text-amber-400 mt-0.5">•</span>
                {note}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

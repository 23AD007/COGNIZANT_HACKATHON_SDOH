import { useEffect, useState } from "react";
import { Activity, AlertTriangle, Brain, Users } from "lucide-react";
import StatCard from "../components/StatCard";
import { api } from "../services/api";

function Dashboard() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api.dashboard().then(setSummary).catch(() => setError("Unable to load the live dashboard."));
  }, []);

  if (error) return <div className="empty-state">{error}</div>;
  if (!summary) return <div className="empty-state">Loading current member data…</div>;

  const distribution = Object.entries(summary.risk_band_counts || {});
  return <div className="dashboard">
    <div className="page-header"><div><h1>SDOH Overview</h1><p>Current member-risk and social determinants intelligence.</p></div><div className="dashboard-status"><span className="status-dot" />Live API</div></div>
    <div className="stats-grid">
      <StatCard title="Total Members" value={summary.member_count.toLocaleString()} subtitle="Current processed artifacts" icon={<Users size={24} />} />
      <StatCard title="Risk Outputs" value={summary.risk_output_member_count.toLocaleString()} subtitle="LightGBM predictions" icon={<Activity size={24} />} />
      <StatCard title="Very High Risk" value={(summary.risk_band_counts?.["Very High"] || 0).toLocaleString()} subtitle="Require priority attention" icon={<AlertTriangle size={24} />} />
      <StatCard title="Recommendation Coverage" value={summary.recommendation_member_count.toLocaleString()} subtitle="Current LambdaMART artifacts" icon={<Brain size={24} />} />
    </div>
    <div className="dashboard-card"><div className="card-header"><div><h2>Risk Distribution</h2><p>Calculated from the current member-risk model.</p></div></div><div className="intervention-list">{distribution.map(([band, count]) => <div className="intervention-row" key={band}><div className="intervention-info"><strong>{band}</strong><span>Current members</span></div><div className="priority-score">{count.toLocaleString()}</div></div>)}</div></div>
  </div>;
}

export default Dashboard;

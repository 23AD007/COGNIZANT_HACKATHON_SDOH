import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Activity, AlertTriangle, Brain, Users } from "lucide-react";
import StatCard from "../components/StatCard";
import { api } from "../services/api";

function Dashboard() {
  const [state, setState] = useState({ loading: true, health: null, members: null, counties: null, locations: null, errors: {} });

  useEffect(() => {
    let active = true;
    const calls = { health: api.health(), members: api.members(), counties: api.getCounties(), locations: api.getCountyLocations() };
    Promise.allSettled(Object.values(calls)).then((results) => {
      if (!active) return;
      const next = { loading: false, health: null, members: null, counties: null, locations: null, errors: {} };
      Object.keys(calls).forEach((key, index) => {
        const result = results[index];
        if (result.status === "fulfilled") next[key] = result.value;
        else next.errors[key] = result.reason?.message || `Unable to load ${key}.`;
      });
      setState(next);
    });
    return () => { active = false; };
  }, []);

  if (state.loading) return <div className="empty-state">Loading current member and county data…</div>;

  const members = Array.isArray(state.members) ? state.members : [];
  const counties = Array.isArray(state.counties) ? state.counties : [];
  const locations = Array.isArray(state.locations) ? state.locations : [];
  const totalCountyMembers = counties.reduce((total, county) => total + (Number(county.member_count) || 0), 0);
  const errors = Object.values(state.errors);

  return <div className="dashboard">
    <div className="page-header"><div><h1>SDOH Overview</h1><p>Current member and county records from the live artifact-backed API.</p></div><div className="dashboard-status"><span className="status-dot" />{state.health?.status || "API status unavailable"}</div></div>
    {errors.length ? <div className="county-message county-message-error"><h2>Some dashboard data is unavailable</h2>{errors.map((message) => <p key={message}>{message}</p>)}</div> : null}
    <div className="stats-grid">
      <StatCard title="Member records" value={members.length.toLocaleString()} subtitle="Returned by /members" icon={<Users size={24} />} />
      <StatCard title="County records" value={counties.length.toLocaleString()} subtitle="Returned by /counties" icon={<Activity size={24} />} />
      <StatCard title="Mapped counties" value={locations.length.toLocaleString()} subtitle="Real-coordinate county groups" icon={<AlertTriangle size={24} />} />
      <StatCard title="County member total" value={totalCountyMembers.toLocaleString()} subtitle="Sum of returned county member counts" icon={<Brain size={24} />} />
    </div>
    <div className="dashboard-card"><div className="card-header"><div><h2>Explore live demo data</h2><p>Member details include individual risk, SDOH, location, and generated recommendations. County details include population priorities.</p></div></div><div className="intervention-list"><div className="intervention-row"><div className="intervention-info"><strong>Members</strong><span>{members.length ? "Open a live member record" : "Member records are currently unavailable"}</span></div><Link className="view-member-button" to="/members">View members</Link></div><div className="intervention-row"><div className="intervention-info"><strong>County Risk Map</strong><span>{locations.length ? "Open real member-location county groups" : "County locations are currently unavailable"}</span></div><Link className="view-member-button" to="/county-map">View counties</Link></div></div></div>
  </div>;
}

export default Dashboard;

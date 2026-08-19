import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Brain, HeartPulse, Lightbulb, MapPin, TrendingUp, UserRound } from "lucide-react";
import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import RiskBadge from "../components/RiskBadge";
import { api } from "../services/api";

const display = (value) => value === null || value === undefined || value === "" ? "Not available" : String(value);
const label = (key) => key.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
const entries = (record, excluded = []) => Object.entries(record || {}).filter(([key, value]) => !excluded.includes(key) && value !== null && value !== undefined && typeof value !== "object");

function ErrorCard({ title, memberId }) { return <section className="detail-card section-error"><h2>{title}</h2><p>Unable to load {title.toLowerCase()}.</p><small>Member ID: {memberId}</small></section>; }
function FieldList({ data, excluded }) { const fields = entries(data, excluded); return fields.length ? <dl className="field-list">{fields.map(([key, value]) => <div key={key}><dt>{label(key)}</dt><dd>{display(value)}</dd></div>)}</dl> : <p className="not-available">Not available</p>; }
function MemberMap({ location }) {
  const lat = Number(location?.lat);
  const lon = Number(location?.lon);
  const hasCoordinates = location?.lat !== null && location?.lat !== undefined && location?.lon !== null && location?.lon !== undefined && Number.isFinite(lat) && Number.isFinite(lon);
  const place = [location?.city, location?.state, location?.county].filter(Boolean).join(", ");
  return <section className="detail-card member-map-card"><div className="detail-card-header"><div><h2>Member Map</h2><p>Selected member's live coordinates</p></div><MapPin size={20} /></div>{hasCoordinates ? <MapContainer key={`${location.member_id}-${lat}-${lon}`} center={[lat, lon]} zoom={12} scrollWheelZoom={false} className="member-map" aria-label="Selected member location map"><TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" /><CircleMarker center={[lat, lon]} radius={9} pathOptions={{ color: "#1d4ed8", fillColor: "#2563eb", fillOpacity: 0.85 }}><Popup>{place || "Selected member location"}</Popup></CircleMarker></MapContainer> : <div className="map-empty-state"><MapPin size={22} /><p>Location coordinates are not available for this member.</p></div>}{place && <p className="map-place-label">{place}</p>}</section>;
}

function MemberDetails() {
  const { memberId } = useParams();
  const [state, setState] = useState({ loading: true, memberId: null, sections: {}, member: null, error: "" });

  useEffect(() => {
    let active = true;
    const calls = { member: api.member(memberId), risk: api.risk(memberId), sdoh: api.sdoh(memberId), clinical: api.clinical(memberId), location: api.location(memberId) };
    const recommendationCall = api.recommendations(memberId);
    Promise.allSettled(Object.values(calls)).then((results) => {
      if (!active) return;
      const sections = Object.fromEntries(Object.keys(calls).map((key, index) => [key, results[index]]));
      sections.recommendations = { status: "pending" };
      const memberResult = sections.member;
      setState({ loading: false, memberId, sections, member: memberResult.status === "fulfilled" ? memberResult.value : null, error: memberResult.status === "rejected" ? (memberResult.reason?.status === 404 ? "Member not found." : "Unable to load this member from the live API.") : "" });
      recommendationCall.then((recommendations) => {
        if (!active) return;
        setState((current) => current.memberId === memberId ? { ...current, sections: { ...current.sections, recommendations: { status: "fulfilled", value: recommendations } } } : current);
      }).catch((reason) => {
        if (!active) return;
        setState((current) => current.memberId === memberId ? { ...current, sections: { ...current.sections, recommendations: { status: "rejected", reason } } } : current);
      });
    });
    return () => { active = false; };
  }, [memberId]);

  if (state.memberId === memberId && state.error) return <div className="empty-member"><h2>{state.error}</h2><p>Member ID: {memberId}</p><Link to="/members">Return to Member Explorer</Link></div>;
  if (state.loading || state.memberId !== memberId || !state.member) return <div className="empty-member loading-state"><span className="loading-dot" />Loading member {memberId}…</div>;

  const { member, sections } = state;
  const section = (name) => sections[name];
  const value = (name) => section(name)?.status === "fulfilled" ? section(name).value : null;
  const risk = value("risk");
  const location = value("location");
  const recommendations = value("recommendations");
  const recommendationList = Array.isArray(recommendations?.recommendations?.recommendations) ? recommendations.recommendations.recommendations : Array.isArray(recommendations?.recommendations) ? recommendations.recommendations : Array.isArray(recommendations) ? recommendations : [];

  return <div className="member-details-page">
    <div className="member-detail-header"><div><Link to="/members" className="back-link"><ArrowLeft size={16} />Back to Members</Link><p className="eyebrow">Selected live member</p><div className="member-title"><div><h1>Member {member.member_id}</h1><p>{[member.age && `Age ${member.age}`, member.gender, member.race, member.ethnicity].filter(Boolean).join(" · ") || "Demographic information not available"}</p></div>{risk?.risk_band ? <RiskBadge level={risk.risk_band} /> : <span className="status-unavailable">Risk status not available</span>}</div></div><div className="member-location"><MapPin size={17} /><span>{[location?.city ?? member.city, location?.state ?? member.state].filter(Boolean).join(", ") || "Location not available"}</span></div></div>
    <div className="member-summary"><UserRound size={18} /><span>Member ID</span><strong>{member.member_id}</strong><span>Selected from live member records</span></div>
    <div className="detail-grid">
      {section("risk")?.status === "rejected" ? <ErrorCard title="Risk Overview" memberId={memberId} /> : <section className="detail-card risk-overview"><div className="detail-card-header"><div><h2>Risk Overview</h2><p>Live model output</p></div><TrendingUp size={20} /></div><div className="risk-score-large"><strong>{risk?.risk_probability !== undefined && risk?.risk_probability !== null ? `${Math.round(Number(risk.risk_probability) * 100)}%` : "Not available"}</strong><span>Risk probability</span></div>{risk?.risk_band ? <RiskBadge level={risk.risk_band} /> : null}<FieldList data={risk} excluded={["member_id", "risk_probability", "risk_band"]} /></section>}
      {section("location")?.status === "rejected" ? <ErrorCard title="Location" memberId={memberId} /> : <section className="detail-card"><div className="detail-card-header"><div><h2>Location</h2><p>Live geographic record</p></div><MapPin size={20} /></div><FieldList data={location} excluded={["member_id"]} /></section>}
      {section("location")?.status === "rejected" ? null : <MemberMap location={location} />}
      {section("sdoh")?.status === "rejected" ? <ErrorCard title="SDOH Context" memberId={memberId} /> : <section className="detail-card"><div className="detail-card-header"><div><h2>SDOH Context</h2><p>Live social determinants data</p></div><Brain size={20} /></div><FieldList data={value("sdoh")} excluded={["member_id"]} /></section>}
      {section("clinical")?.status === "rejected" ? <ErrorCard title="Clinical Context" memberId={memberId} /> : <section className="detail-card"><div className="detail-card-header"><div><h2>Clinical Context</h2><p>Live clinical record</p></div><HeartPulse size={20} /></div><FieldList data={value("clinical")} excluded={["member_id"]} /></section>}
    </div>
    {section("recommendations")?.status === "pending" ? <section className="detail-card recommendations-card section-loading"><div className="detail-card-header"><div><h2>Member Recommendations</h2><p>Generating recommendations for member ID {member.member_id}…</p></div><Lightbulb size={20} /></div><div className="loading-lines"><i /><i /><i /></div></section> : section("recommendations")?.status === "rejected" ? <section className="detail-card recommendations-card section-error"><div className="detail-card-header"><div><h2>Member Recommendations</h2><p>Recommendations for member ID {member.member_id}</p></div><Lightbulb size={20} /></div><p>{section("recommendations").reason?.message || "Unable to load member recommendations."}</p><small>Member ID: {member.member_id}</small></section> : <section className="detail-card recommendations-card"><div className="detail-card-header"><div><h2>Member Recommendations</h2><p>Live recommendations for member ID {member.member_id}</p></div><Lightbulb size={20} /></div>{recommendationList.length ? <div className="personalized-interventions">{recommendationList.map((item, index) => <article className="personalized-intervention" key={item.intervention_id || item.id || index}><span className="intervention-rank">{index + 1}</span><div className="personalized-info"><strong>{display(item.intervention_name ?? item.name ?? item.action ?? item.recommendation)}</strong><FieldList data={item} excluded={["intervention_id", "id", "intervention_name", "name", "action", "recommendation"]} /></div></article>)}</div> : <div className="recommendation-unavailable"><p>Member recommendations are not available for this member.</p><small>Member ID: {member.member_id}</small></div>}</section>}
  </div>;
}

export default MemberDetails;

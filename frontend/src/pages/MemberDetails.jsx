import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, Brain, MapPin, TrendingUp } from "lucide-react";
import RiskBadge from "../components/RiskBadge";
import { api } from "../services/api";

function MemberDetails() {
  const { memberId } = useParams();
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  useEffect(() => {
    Promise.all([api.member(memberId), api.risk(memberId), api.sdoh(memberId), api.clinical(memberId), api.location(memberId), api.recommendations(memberId).catch(() => null)])
      .then(([member, risk, sdoh, clinical, location, recommendations]) => setData({ member, risk, sdoh, clinical, location, recommendations }))
      .catch((requestError) => setError(requestError.status === 404 ? "Member not found" : "Unable to load member details."));
  }, [memberId]);
  if (error) return <div className="empty-member"><h2>{error}</h2><Link to="/members">Return to Member Explorer</Link></div>;
  if (!data) return <div className="empty-member">Loading current member data…</div>;
  const { member, risk, sdoh, clinical, location, recommendations } = data;
  const sdohFields = Object.entries(sdoh).filter(([key, value]) => key !== "member_id" && typeof value === "number").slice(0, 12);
  const recommendationList = recommendations?.recommendations?.recommendations || recommendations?.recommendations || [];
  return <div className="member-details-page"><div className="member-detail-header"><div><Link to="/members" className="back-link"><ArrowLeft size={16} />Back to Members</Link><div className="member-title"><div><h1>Member {member.member_id}</h1><p>Current SDOH risk profile</p></div><RiskBadge level={risk.risk_band} /></div></div><div className="member-location"><MapPin size={17} /><span>{location.county || location.city || "Location unavailable"}</span></div></div><div className="detail-grid"><div className="detail-card risk-overview"><div className="detail-card-header"><div><h2>Risk Overview</h2><p>Model-generated current risk</p></div><TrendingUp size={20} /></div><div className="risk-score-large"><strong>{Math.round(risk.risk_probability * 100)}%</strong><span>Overall Risk Score</span></div><RiskBadge level={risk.risk_band} /></div><div className="detail-card"><div className="detail-card-header"><div><h2>County Context</h2><p>Current geographic record</p></div><MapPin size={20} /></div><div className="county-context"><div><span>County</span><strong>{location.county || "—"}</strong></div><div><span>County FIPS</span><strong>{location.county_fips || "—"}</strong></div><div><span>City</span><strong>{location.city || "—"}</strong></div><div><span>State</span><strong>{location.state || "—"}</strong></div></div></div></div><div className="detail-card"><div className="detail-card-header"><div><h2>SDOH and Clinical Features</h2><p>Current artifacts for this member</p></div><Brain size={20} /></div><div className="sdoh-driver-list">{sdohFields.map(([name, value]) => <div className="sdoh-driver" key={name}><div className="driver-name">{name}</div><strong>{Number(value).toFixed(2)}</strong></div>)}<div className="sdoh-driver"><div className="driver-name">Clinical conditions</div><strong>{clinical.clinical_condition_count ?? "—"}</strong></div></div></div><div className="detail-card"><div className="detail-card-header"><div><h2>Personalized Intervention Priorities</h2><p>Current LambdaMART recommendations, when available</p></div><TrendingUp size={20} /></div>{recommendationList.length ? <div className="personalized-interventions">{recommendationList.map((item, index) => <div className="personalized-intervention" key={item.intervention_id || index}><div className="intervention-rank">#{item.rank || index + 1}</div><div className="personalized-info"><strong>{item.intervention_name || item.name}</strong><p>{item.rationale || item.action}</p></div></div>)}</div> : <p>No current recommendation artifact is available for this member.</p>}</div></div>;
}
export default MemberDetails;

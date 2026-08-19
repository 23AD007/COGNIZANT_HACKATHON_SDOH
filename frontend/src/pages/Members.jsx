import { useEffect, useMemo, useState } from "react";
import { Eye, Search, Users } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../services/api";

const display = (value) => value ?? "Not available";

function Members() {
  const [members, setMembers] = useState([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    api.members()
      .then((response) => setMembers(Array.isArray(response) ? response : []))
      .catch(() => setError("Unable to load members from the live API."))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return members;
    return members.filter((member) => [member.member_id, member.age, member.gender, member.race, member.ethnicity, member.city, member.state]
      .some((value) => String(value ?? "").toLowerCase().includes(query)));
  }, [members, search]);

  if (loading) return <div className="empty-state loading-state"><span className="loading-dot" />Loading current members from the live API…</div>;
  if (error) return <div className="empty-state">{error}</div>;

  return <div className="members-page">
    <div className="page-header member-page-heading"><div><p className="eyebrow">Member flow</p><h1>Member Explorer</h1><p>Select a live member to review their risk, context, and available recommendations.</p></div><div className="member-count"><Users size={19} /><strong>{members.length.toLocaleString()}</strong><span>live members</span></div></div>
    <div className="filter-card"><div className="member-search"><Search size={18} /><input aria-label="Search members" placeholder="Search ID, demographics, city, or state…" value={search} onChange={(event) => setSearch(event.target.value)} /></div></div>
    <div className="member-table-card"><div className="table-header"><div><h2>Current Members</h2><p>{filtered.length.toLocaleString()} of {members.length.toLocaleString()} live member records</p></div></div><div className="table-wrapper"><table className="member-table"><thead><tr><th>Member ID</th><th>Age</th><th>Gender</th><th>Race</th><th>Ethnicity</th><th>Location</th><th>Action</th></tr></thead><tbody>{filtered.map((member) => <tr key={member.member_id}><td><strong>{display(member.member_id)}</strong></td><td>{display(member.age)}</td><td>{display(member.gender)}</td><td>{display(member.race)}</td><td>{display(member.ethnicity)}</td><td>{[member.city, member.state].filter(Boolean).join(", ") || "Not available"}</td><td><Link to={`/members/${encodeURIComponent(member.member_id)}`} className="view-member-button"><Eye size={16} />View member</Link></td></tr>)}</tbody></table>{!filtered.length && <div className="table-empty">No live members match this search.</div>}</div></div>
  </div>;
}

export default Members;

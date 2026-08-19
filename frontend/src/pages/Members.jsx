import { useEffect, useMemo, useState } from "react";
import { Eye, Search } from "lucide-react";
import { Link } from "react-router-dom";
import { api } from "../services/api";

function Members() {
  const [members, setMembers] = useState([]);
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");
  useEffect(() => { api.members().then(setMembers).catch(() => setError("Unable to load current members.")); }, []);
  const filtered = useMemo(() => members.filter((member) => member.member_id.toLowerCase().includes(search.toLowerCase())), [members, search]);
  return <div className="members-page"><div className="page-header"><div><h1>Member Explorer</h1><p>Search current member-risk records.</p></div></div><div className="filter-card"><div className="member-search"><Search size={18} /><input placeholder="Search Member ID..." value={search} onChange={(event) => setSearch(event.target.value)} /></div></div>{error ? <div className="empty-state">{error}</div> : <div className="member-table-card"><div className="table-header"><div><h2>Members</h2><p>{filtered.length.toLocaleString()} members found</p></div></div><div className="table-wrapper"><table className="member-table"><thead><tr><th>Member ID</th><th>Age</th><th>City</th><th>State</th><th>Action</th></tr></thead><tbody>{filtered.map((member) => <tr key={member.member_id}><td><strong>{member.member_id}</strong></td><td>{member.age ?? "—"}</td><td>{member.city ?? "—"}</td><td>{member.state ?? "—"}</td><td><Link to={`/members/${member.member_id}`} className="view-member-button"><Eye size={16} />View</Link></td></tr>)}</tbody></table></div></div>}</div>;
}
export default Members;

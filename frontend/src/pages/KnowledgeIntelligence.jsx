import { useEffect, useMemo, useState } from "react";
import { BookOpen, Network, Search } from "lucide-react";
import { api } from "../services/api";

const display = (value) => value == null || value === "" ? "Not available" : String(value);
const label = (value) => value.replace(/_/g, " ").replace(/\b\w/g, (letter) => letter.toUpperCase());

function Properties({ values }) {
  const entries = Object.entries(values || {});
  return entries.length ? <dl className="knowledge-properties">{entries.map(([key, value]) => <div key={key}><dt>{label(key)}</dt><dd>{display(value)}</dd></div>)}</dl> : null;
}

function KnowledgeIntelligence() {
  const [state, setState] = useState({ loading: true, data: null, error: "" });
  const [filter, setFilter] = useState("");

  useEffect(() => {
    let active = true;
    api.knowledge()
      .then((data) => { if (active) setState({ loading: false, data, error: "" }); })
      .catch((error) => { if (active) setState({ loading: false, data: null, error: error.message || "Unable to load Knowledge Graph." }); });
    return () => { active = false; };
  }, []);

  const content = useMemo(() => {
    const query = filter.trim().toLowerCase();
    if (!state.data || !query) return { nodes: state.data?.nodes || [], relationships: state.data?.relationships || [] };
    const matches = (record) => JSON.stringify(record).toLowerCase().includes(query);
    return { nodes: state.data.nodes.filter(matches), relationships: state.data.relationships.filter(matches) };
  }, [filter, state.data]);

  if (state.loading) return <div className="empty-state loading-state"><span className="loading-dot" />Loading the persisted Knowledge Graph…</div>;
  if (state.error) return <div className="county-message county-message-error"><h2>Knowledge Graph unavailable</h2><p>{state.error}</p></div>;

  const { data } = state;
  const hasRecords = data.nodes.length || data.relationships.length;
  return <div className="knowledge-page">
    <div className="page-header"><div><h1>Knowledge Intelligence</h1><p>Read-only concepts and relationships from the persisted Knowledge Graph.</p></div><div className="knowledge-status"><BookOpen size={17} />Knowledge Graph</div></div>
    <section className="stats-grid knowledge-stats" aria-label="Knowledge Graph summary">
      <div className="stat-card"><Network size={24} /><div><p>Graph nodes</p><h2>{data.node_count.toLocaleString()}</h2><span>All artifact node types</span></div></div>
      <div className="stat-card"><Network size={24} /><div><p>Graph relationships</p><h2>{data.relationship_count.toLocaleString()}</h2><span>All artifact relationship types</span></div></div>
      <div className="stat-card"><BookOpen size={24} /><div><p>Displayed concepts</p><h2>{data.nodes.length.toLocaleString()}</h2><span>Domains, factors, evidence, interventions</span></div></div>
    </section>
    <section className="dashboard-card knowledge-summary"><div className="card-header"><div><h2>Artifact summary</h2><p>Schema version {display(data.schema_version)}</p></div></div><div className="knowledge-counts">{Object.entries(data.node_types).map(([type, count]) => <span key={type}>{type}: {count}</span>)}</div><div className="knowledge-counts">{Object.entries(data.relationship_types).map(([type, count]) => <span key={type}>{type}: {count}</span>)}</div></section>
    <section className="dashboard-card"><div className="card-header"><div><h2>Knowledge Graph records</h2><p>Search matches only fields returned by the graph artifact.</p></div></div><label className="knowledge-search"><Search size={17} /><span className="sr-only">Search graph records</span><input value={filter} onChange={(event) => setFilter(event.target.value)} placeholder="Search type, label, source, factor, or relationship" /></label></section>
    {!hasRecords ? <div className="county-message"><h2>No usable Knowledge Graph records are available.</h2><p>The artifact returned no exposed domains, factors, evidence, interventions, or relationships.</p></div> : <div className="knowledge-grid">
      <section className="dashboard-card"><div className="card-header"><div><h2>Concepts</h2><p>{content.nodes.length} returned records</p></div></div><div className="knowledge-record-list">{content.nodes.map((node) => <article className="knowledge-record" key={node.node_id}><span className="knowledge-type">{node.node_type}</span><h3>{display(node.label)}</h3><Properties values={node.properties} /></article>)}{!content.nodes.length ? <p className="not-available">No graph concepts match this search.</p> : null}</div></section>
      <section className="dashboard-card"><div className="card-header"><div><h2>Relationships</h2><p>{content.relationships.length} returned relationships</p></div></div><div className="knowledge-record-list">{content.relationships.map((relationship) => <article className="knowledge-record" key={relationship.edge_id || `${relationship.source}-${relationship.target}-${relationship.relationship_type}`}><span className="knowledge-type">{relationship.relationship_type}</span><p className="knowledge-edge">{relationship.source} → {relationship.target}</p><Properties values={relationship.properties} /></article>)}{!content.relationships.length ? <p className="not-available">No graph relationships match this search.</p> : null}</div></section>
    </div>}
  </div>;
}

export default KnowledgeIntelligence;

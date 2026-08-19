import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Building2, CheckCircle2, MapPinned, Users } from "lucide-react";
import { CircleMarker, MapContainer, Popup, TileLayer } from "react-leaflet";
import { getCounties, getCountyLocations } from "../services/api";

const unavailable = (value) => value == null || value === "" ? "Not available" : value;

function CountyRiskMap() {
  const [counties, setCounties] = useState([]);
  const [error, setError] = useState("");
  const [selectedFips, setSelectedFips] = useState("");
  const [locations, setLocations] = useState([]);
  const [locationsLoading, setLocationsLoading] = useState(true);
  const [locationError, setLocationError] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    getCounties().then(setCounties).catch(() => setError("Unable to load county records from the live API."));
    getCountyLocations()
      .then(setLocations)
      .catch((requestError) => setLocationError(requestError.message || "Unable to load real county locations."))
      .finally(() => setLocationsLoading(false));
  }, []);
  const overview = useMemo(() => ({
    members: counties.reduce((total, county) => total + (Number(county.member_count) || 0), 0),
    coverage: counties.length ? counties.reduce((total, county) => total + (Number(county.risk_coverage) || 0), 0) / counties.length : null,
    priorities: counties.filter((county) => county.recommendation_available).length,
  }), [counties]);
  const countyByFips = useMemo(() => new Map(counties.map((county) => [county.county_fips, county])), [counties]);
  const mappedCounties = useMemo(() => locations.filter((location) => countyByFips.has(location.county_fips)), [countyByFips, locations]);
  const selectCounty = (countyFips) => { setSelectedFips(countyFips); navigate(`/counties/${countyFips}`); };

  return <div className="county-page">
    <div className="page-header"><div><h1>County Risk Overview</h1><p>Live member-population risk coverage and county selection.</p></div><div className="map-status"><MapPinned size={17} />Live county API</div></div>
    {error ? <div className="county-message county-message-error"><h2>County records unavailable</h2><p>{error}</p></div> : !counties.length ? <div className="county-message"><h2>Loading county risk overview…</h2><p>Retrieving current counties and member-population coverage.</p></div> : <>
      <section className="county-overview-grid" aria-label="County risk overview">
        <div className="county-overview-card"><Building2 size={21} /><div><span>Counties represented</span><strong>{counties.length.toLocaleString()}</strong><small>Current county records</small></div></div>
        <div className="county-overview-card"><Users size={21} /><div><span>Members represented</span><strong>{overview.members.toLocaleString()}</strong><small>From returned county counts</small></div></div>
        <div className="county-overview-card"><CheckCircle2 size={21} /><div><span>Average risk coverage</span><strong>{overview.coverage == null ? "Not available" : `${Math.round(overview.coverage * 100)}%`}</strong><small>Across returned counties</small></div></div>
        <div className="county-overview-card"><MapPinned size={21} /><div><span>Priority availability</span><strong>{overview.priorities.toLocaleString()}</strong><small>Counties reporting priorities</small></div></div>
      </section>
      <section className="dashboard-card county-map-card"><div className="card-header"><div><h2>County Risk Map</h2><p>Real member locations aggregated by county FIPS. Select a marker to open County Details.</p></div><span className="county-map-count">{mappedCounties.length} mapped counties</span></div>{locationsLoading ? <div className="county-message"><h3>Loading county locations</h3><p>Loading real member coordinates by county.</p></div> : mappedCounties.length ? <MapContainer center={[mappedCounties[0].lat, mappedCounties[0].lon]} zoom={7} scrollWheelZoom={false} className="county-risk-map" aria-label="County risk map"><TileLayer attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors' url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" />{mappedCounties.map((location) => { const county = countyByFips.get(location.county_fips); const radius = Math.max(7, Math.min(22, Math.sqrt(location.mapped_member_count) * 2)); return <CircleMarker key={location.county_fips} center={[location.lat, location.lon]} radius={radius} pathOptions={{ color: selectedFips === location.county_fips ? "#1d4ed8" : "#0369a1", fillColor: selectedFips === location.county_fips ? "#2563eb" : "#38bdf8", fillOpacity: 0.72 }} eventHandlers={{ click: () => selectCounty(location.county_fips) }}><Popup><strong>{unavailable(county.county_name)}, {unavailable(county.state_abbr)}</strong><br />FIPS: {location.county_fips}<br />Mapped members: {location.mapped_member_count}<br />Risk coverage: {county.risk_coverage == null ? "Not available" : `${Math.round(county.risk_coverage * 100)}%`}</Popup></CircleMarker>; })}</MapContainer> : <div className="county-message"><h3>County locations unavailable</h3><p>{locationError || "No current county has usable member coordinates."}</p></div>}</section>
      <section className="dashboard-card county-list-card"><div className="card-header"><div><h2>Select a county</h2><p>Choose a real county record to review its risk context and population priorities.</p></div></div><div className="table-wrapper"><table className="member-table county-table"><thead><tr><th>County</th><th>State</th><th>FIPS</th><th>Members</th><th>Risk coverage</th><th /></tr></thead><tbody>{counties.map((county) => <tr key={county.county_fips} className={selectedFips === county.county_fips ? "county-row-selected" : ""}><td><strong>{unavailable(county.county_name)}</strong></td><td>{unavailable(county.state_abbr)}</td><td>{county.county_fips}</td><td>{county.member_count == null ? "Not available" : Number(county.member_count).toLocaleString()}</td><td>{county.risk_coverage == null ? "Not available" : `${Math.round(county.risk_coverage * 100)}%`}</td><td><Link className="view-member-button" to={`/counties/${county.county_fips}`} onClick={() => setSelectedFips(county.county_fips)} aria-current={selectedFips === county.county_fips ? "page" : undefined}>View county</Link></td></tr>)}</tbody></table></div></section>
    </>}
  </div>;
}

export default CountyRiskMap;

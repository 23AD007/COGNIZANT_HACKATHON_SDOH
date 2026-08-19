import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft, MapPin } from "lucide-react";
import { getCounty, getCountyRecommendations } from "../services/api";

const contextFields = ["poverty_pct", "unemployment_pct", "uninsured_pct", "housing_no_vehicle_pct", "median_household_income"];
const formatValue = (value) => value == null || value === "" ? "Not available" : typeof value === "number" ? value.toFixed(2) : value;

function CountyDetails() {
  const { countyFips } = useParams();
  const [countyResult, setCountyResult] = useState(null);
  const [prioritiesResult, setPrioritiesResult] = useState(null);

  useEffect(() => {
    getCounty(countyFips).then((county) => setCountyResult({ fips: countyFips, county })).catch((requestError) => setCountyResult({ fips: countyFips, error: requestError.status === 404 ? `County FIPS ${countyFips} was not found.` : `Unable to load county FIPS ${countyFips} (API returned ${requestError.status || "an error"}).` }));
    getCountyRecommendations(countyFips).then((payload) => setPrioritiesResult({ fips: countyFips, priorities: payload.recommendations || [] })).catch((requestError) => setPrioritiesResult({ fips: countyFips, error: `Unable to load priorities for county FIPS ${countyFips} (API returned ${requestError.status || "an error"}).` }));
  }, [countyFips]);

  const countyError = countyResult?.fips === countyFips ? countyResult.error : null;
  const county = countyResult?.fips === countyFips ? countyResult.county : null;
  const prioritiesError = prioritiesResult?.fips === countyFips ? prioritiesResult.error : null;
  const priorities = prioritiesResult?.fips === countyFips ? prioritiesResult.priorities : null;
  if (countyError) return <div className="county-message county-message-error"><h2>County details unavailable</h2><p>{countyError}</p><Link to="/county-map">Return to County Risk Overview</Link></div>;
  if (!county) return <div className="county-message"><h2>Loading county FIPS {countyFips}…</h2><p>Retrieving live county risk and SDOH context.</p></div>;

  const context = county.sdoh_features || county.county_features || {};
  return <div className="member-details-page"><div className="member-detail-header"><div><Link to="/county-map" className="back-link"><ArrowLeft size={16} />Back to County Risk Overview</Link><div className="member-title"><div><h1>{formatValue(county.county_name)}, {formatValue(county.state_abbr)}</h1><p>Selected county FIPS {county.county_fips}</p></div></div></div><div className="member-location"><MapPin size={17} /><span>{formatValue(county.state_name)}</span></div></div><div className="detail-grid"><div className="detail-card risk-overview"><div className="detail-card-header"><div><h2>Population Risk</h2><p>Aggregate of current county members</p></div></div><div className="risk-score-large"><strong>{county.mean_risk == null ? "Not available" : `${Math.round(county.mean_risk * 100)}%`}</strong><span>{formatValue(county.risk_band)}</span></div><p>{formatValue(county.risk_member_count)} of {formatValue(county.member_count)} members have valid risk results.</p></div><div className="detail-card"><div className="detail-card-header"><div><h2>Risk Distribution</h2><p>Returned population risk bands</p></div></div><div className="sdoh-driver-list">{Object.entries(county.risk_distribution || {}).length ? Object.entries(county.risk_distribution).map(([band, count]) => <div className="sdoh-driver" key={band}><div className="driver-name">{band}</div><strong>{count}</strong></div>) : <p>Not available</p>}</div></div><div className="detail-card"><div className="detail-card-header"><div><h2>County SDOH Context</h2><p>County fields returned by the API</p></div></div><div className="sdoh-driver-list">{contextFields.filter((field) => context[field] != null).length ? contextFields.filter((field) => context[field] != null).map((field) => <div className="sdoh-driver" key={field}><div className="driver-name">{field}</div><strong>{formatValue(context[field])}</strong></div>) : <p>Not available</p>}</div></div><section className="detail-card county-priorities-card"><div className="detail-card-header"><div><h2>County Population Priorities</h2><p>Population-level priorities derived from current county members</p></div></div>{prioritiesError ? <div className="county-message county-message-error"><h3>County priorities unavailable</h3><p>{prioritiesError}</p></div> : priorities === null ? <div className="county-message"><h3>Loading county priorities…</h3><p>Retrieving priorities for county FIPS {county.county_fips}.</p></div> : priorities.length ? <div className="county-priority-list">{priorities.map((item) => <article className="county-priority-card" key={`${item.rank}-${item.intervention}`}><div className="county-priority-rank">#{formatValue(item.rank)}</div><div><h3>{formatValue(item.intervention)}</h3><p>{formatValue(item.rationale)}</p><div className="county-priority-meta">{item.domain != null && <span>Domain: {item.domain}</span>}{item.affected_member_count != null && <span>Affected members: {item.affected_member_count}</span>}{item.prevalence != null && <span>Prevalence: {Math.round(item.prevalence * 100)}%</span>}</div></div></article>)}</div> : <div className="county-message"><h3>No population priorities are currently available for this county.</h3></div>}</section></div></div>;
}

export default CountyDetails;

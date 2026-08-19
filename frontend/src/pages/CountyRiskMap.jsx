import { MapPinned } from "lucide-react";

function CountyRiskMap() {
  return (
    <div className="county-page">
      <div className="page-header">
        <div>
          <h1>County Risk Intelligence</h1>
          <p>County-level information is shown only when a supported project data source is available.</p>
        </div>
        <div className="map-status">
          <MapPinned size={17} />
          County data unavailable
        </div>
      </div>
      <div className="dashboard-card empty-state">
        <h2>County-level recommendations are not currently available.</h2>
        <p>County APIs are not exposed by the current backend, so this view does not display placeholder counties, risk scores, locations, or recommendations.</p>
      </div>
    </div>
  );
}

export default CountyRiskMap;

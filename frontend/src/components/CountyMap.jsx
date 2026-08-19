import {
  MapContainer,
  TileLayer,
  CircleMarker,
  Popup,
} from "react-leaflet";

import { counties } from "../data/mockData";

function getRiskColor(level) {
  if (level === "High") {
    return "#dc2626";
  }

  if (level === "Medium") {
    return "#f59e0b";
  }

  return "#16a34a";
}

function CountyMap({ onCountySelect }) {
  return (
    <MapContainer
      center={[40.7128, -74.0060]}
      zoom={10}
      className="county-map"
    >

      <TileLayer
        attribution='&copy; OpenStreetMap contributors'
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
      />

      {counties.map((county) => (

        <CircleMarker
          key={county.id}
          center={[
            county.latitude,
            county.longitude,
          ]}
          radius={18}
          pathOptions={{
            color: getRiskColor(
              county.riskLevel
            ),
            fillColor: getRiskColor(
              county.riskLevel
            ),
            fillOpacity: 0.65,
          }}
          eventHandlers={{
            click: () => {
              onCountySelect(county);
            },
          }}
        >

          <Popup>

            <div className="map-popup">

              <h3>
                {county.name}
              </h3>

              <p>
                Risk:
                <strong>
                  {" "}
                  {county.riskLevel}
                </strong>
              </p>

              <p>
                Risk Score:
                {" "}
                {Math.round(
                  county.riskScore * 100
                )}
                %
              </p>

              <p>
                Members:
                {" "}
                {county.members.toLocaleString()}
              </p>

              <p>
                Primary SDOH:
                {" "}
                {county.primarySdoh}
              </p>

            </div>

          </Popup>

        </CircleMarker>

      ))}

    </MapContainer>
  );
}

export default CountyMap;
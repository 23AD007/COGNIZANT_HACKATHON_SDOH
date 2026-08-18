import { useState } from "react";

import {
  MapPinned,
  Users,
  AlertTriangle,
  TrendingUp,
} from "lucide-react";

import CountyMap from "../components/CountyMap";

import { counties } from "../data/mockData";

function CountyRiskMap() {

  const [selectedCounty, setSelectedCounty] =
    useState(counties[0]);

  return (
    <div className="county-page">

      {/* Header */}

      <div className="page-header">

        <div>

          <h1>
            County Risk Intelligence
          </h1>

          <p>
            Geographic analysis of SDOH risk
            and intervention priorities.
          </p>

        </div>

        <div className="map-status">

          <MapPinned size={17} />

          Geographic Intelligence

        </div>

      </div>


      {/* Map */}

      <div className="map-card">

        <CountyMap
          onCountySelect={setSelectedCounty}
        />

      </div>


      {/* Selected County */}

      <div className="county-detail-grid">

        {/* County Overview */}

        <div className="dashboard-card">

          <div className="card-header">

            <div>

              <h2>
                {selectedCounty.name}
              </h2>

              <p>
                {selectedCounty.state}
              </p>

            </div>

            <MapPinned size={21} />

          </div>


          <div className="county-risk-number">

            <strong>
              {Math.round(
                selectedCounty.riskScore * 100
              )}
              %
            </strong>

            <span>
              Overall County Risk
            </span>

          </div>


          <div className="county-risk-badge">

            <span
              className={`risk-badge ${selectedCounty.riskLevel.toLowerCase()}`}
            >
              {selectedCounty.riskLevel} Risk
            </span>

          </div>


          <div className="county-stat-row">

            <div>

              <Users size={18} />

              <div>

                <span>
                  Members
                </span>

                <strong>
                  {selectedCounty.members.toLocaleString()}
                </strong>

              </div>

            </div>


            <div>

              <AlertTriangle size={18} />

              <div>

                <span>
                  Primary SDOH
                </span>

                <strong>
                  {selectedCounty.primarySdoh}
                </strong>

              </div>

            </div>

          </div>

        </div>


        {/* SDOH Profile */}

        <div className="dashboard-card">

          <div className="card-header">

            <div>

              <h2>
                County SDOH Profile
              </h2>

              <p>
                Major social risk factors
              </p>

            </div>

            <TrendingUp size={21} />

          </div>


          <div className="county-sdoh-list">

            {selectedCounty.sdohDrivers.map(
              (driver) => (

                <div
                  className="county-sdoh-row"
                  key={driver.name}
                >

                  <div className="county-sdoh-name">
                    {driver.name}
                  </div>

                  <div className="county-sdoh-bar">

                    <div
                      style={{
                        width:
                          `${driver.score}%`,
                      }}
                    />

                  </div>

                  <strong>
                    {driver.score}%
                  </strong>

                </div>

              )
            )}

          </div>

        </div>

      </div>


      {/* Intervention */}

      <div className="dashboard-card">

        <div className="card-header">

          <div>

            <h2>
              Recommended County Intervention
            </h2>

            <p>
              Highest-priority intervention based
              on county SDOH profile
            </p>

          </div>

          <TrendingUp size={21} />

        </div>


        <div className="county-intervention">

          <div className="county-intervention-rank">
            #1
          </div>

          <div>

            <strong>
              {selectedCounty.intervention}
            </strong>

            <p>
              Prioritized based on the dominant
              SDOH risk factor in this county.
            </p>

          </div>

        </div>

      </div>

    </div>
  );
}

export default CountyRiskMap;
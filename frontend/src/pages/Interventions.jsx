import { useState } from "react";
import {
  ArrowUpDown,
  Brain,
  CheckCircle,
  Info,
  TrendingUp,
} from "lucide-react";

import { interventionRanking } from "../data/mockData";

function Interventions() {
  const [selected, setSelected] = useState(
    interventionRanking[0]
  );

  const [sortOrder, setSortOrder] = useState("desc");

  const sortedInterventions = [...interventionRanking].sort(
    (a, b) => {
      return sortOrder === "desc"
        ? b.score - a.score
        : a.score - b.score;
    }
  );

  return (
    <div className="interventions-page">

      {/* Header */}

      <div className="page-header">

        <div>
          <h1>Intervention Ranking</h1>

          <p>
            Prioritized interventions based on SDOH
            risk and member needs.
          </p>
        </div>

        <div className="model-status">
          <Brain size={17} />
          LambdaMART Ranking
        </div>

      </div>


      {/* Summary */}

      <div className="intervention-summary">

        <div className="summary-card">

          <TrendingUp size={21} />

          <div>
            <span>Ranked Interventions</span>
            <strong>
              {interventionRanking.length}
            </strong>
          </div>

        </div>


        <div className="summary-card">

          <CheckCircle size={21} />

          <div>
            <span>Top Priority</span>
            <strong>
              {interventionRanking[0].intervention}
            </strong>
          </div>

        </div>


        <div className="summary-card">

          <Info size={21} />

          <div>
            <span>Top Ranking Score</span>
            <strong>
              {Math.round(
                interventionRanking[0].score * 100
              )}
              %
            </strong>
          </div>

        </div>

      </div>


      {/* Main Content */}

      <div className="intervention-layout">

        {/* Ranking List */}

        <div className="dashboard-card">

          <div className="card-header">

            <div>
              <h2>Priority Ranking</h2>

              <p>
                Higher scores indicate higher
                intervention priority.
              </p>
            </div>

            <button
              className="sort-button"
              onClick={() =>
                setSortOrder(
                  sortOrder === "desc"
                    ? "asc"
                    : "desc"
                )
              }
            >
              <ArrowUpDown size={16} />
              Sort
            </button>

          </div>


          <div className="ranking-list">

            {sortedInterventions.map(
              (item) => (

                <button
                  key={item.intervention}
                  className={`ranking-item ${
                    selected.intervention ===
                    item.intervention
                      ? "selected"
                      : ""
                  }`}
                  onClick={() =>
                    setSelected(item)
                  }
                >

                  <div className="ranking-number">
                    #{item.rank}
                  </div>

                  <div className="ranking-main">

                    <strong>
                      {item.intervention}
                    </strong>

                    <span>
                      {item.category}
                    </span>

                  </div>

                  <div className="ranking-score">

                    <strong>
                      {Math.round(
                        item.score * 100
                      )}
                      %
                    </strong>

                    <small>
                      {item.priority}
                    </small>

                  </div>

                </button>

              )
            )}

          </div>

        </div>


        {/* Details */}

        <div className="dashboard-card">

          <div className="card-header">

            <div>
              <h2>
                Intervention Details
              </h2>

              <p>
                Explanation of the selected
                recommendation.
              </p>
            </div>

            <Brain size={21} />

          </div>


          <div className="selected-intervention">

            <div className="selected-rank">
              #{selected.rank}
            </div>

            <div>

              <h2>
                {selected.intervention}
              </h2>

              <span>
                {selected.category}
              </span>

            </div>

          </div>


          <div className="priority-meter">

            <div className="meter-header">

              <span>
                Ranking Score
              </span>

              <strong>
                {Math.round(
                  selected.score * 100
                )}
                %
              </strong>

            </div>

            <div className="meter">

              <div
                style={{
                  width:
                    `${selected.score * 100}%`,
                }}
              />

            </div>

          </div>


          <div className="intervention-detail-section">

            <h3>
              Why was this prioritized?
            </h3>

            <p>
              {selected.reason}
            </p>

          </div>


          <div className="intervention-detail-section">

            <h3>
              Target Population
            </h3>

            <p>
              {selected.target}
            </p>

          </div>


          <div className="intervention-detail-section">

            <h3>
              Potentially Affected Members
            </h3>

            <strong className="affected-count">
              {selected.affectedMembers.toLocaleString()}
            </strong>

          </div>


          <div className="evidence-mini">

            <div className="evidence-mini-header">

              <CheckCircle size={18} />

              <strong>
                Evidence Context
              </strong>

            </div>

            <p>
              {selected.evidence}
            </p>

          </div>

        </div>

      </div>

    </div>
  );
}

export default Interventions;
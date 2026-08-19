import { Brain } from "lucide-react";

function Interventions() {

  return (
    <div className="interventions-page">

      {/* Header */}

      <div className="page-header">

        <div>
          <h1>Intervention Ranking</h1>

          <p>
            Prioritized interventions are shown only when a supported project data source is available.
          </p>
        </div>

        <div className="model-status">
          <Brain size={17} />
          Rankings unavailable
        </div>

      </div>


      <div className="dashboard-card empty-state">
        <h2>Global intervention rankings are not currently available.</h2>
        <p>The current API provides recommendations only for the selected member when a real recommendation artifact exists.</p>
      </div>

    </div>
  );
}

export default Interventions;

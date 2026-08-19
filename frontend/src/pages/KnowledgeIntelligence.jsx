import { BookOpen } from "lucide-react";

function KnowledgeIntelligence() {

  return (
    <div className="knowledge-page">

      {/* Header */}

      <div className="page-header">

        <div>

          <h1>
            Knowledge Intelligence
          </h1>

          <p>
            Evidence is shown only when it is provided by a supported project data source.
          </p>

        </div>

        <div className="knowledge-status">

          <BookOpen size={17} />

          Evidence unavailable

        </div>

      </div>


      <div className="dashboard-card empty-state">
        <h2>Evidence knowledge is not currently available.</h2>
        <p>No evidence API is exposed by the current backend, so this screen does not display placeholder evidence, confidence values, or citations.</p>
      </div>

    </div>
  );
}

export default KnowledgeIntelligence;

import { useState } from "react";

import {
  BookOpen,
  CheckCircle,
  Search,
  ShieldCheck,
  Sparkles,
} from "lucide-react";

import { knowledgeEvidence } from "../data/mockData";

function KnowledgeIntelligence() {

  const [search, setSearch] = useState("");

  const [selected, setSelected] = useState(
    knowledgeEvidence[0]
  );

  const filteredEvidence =
    knowledgeEvidence.filter((item) => {

      const text = `
        ${item.intervention}
        ${item.sdohFactor}
        ${item.source}
        ${item.explanation}
        ${item.tags.join(" ")}
      `.toLowerCase();

      return text.includes(
        search.toLowerCase()
      );
    });

  return (
    <div className="knowledge-page">

      {/* Header */}

      <div className="page-header">

        <div>

          <h1>
            Knowledge Intelligence
          </h1>

          <p>
            Evidence-based context supporting
            SDOH intervention recommendations.
          </p>

        </div>

        <div className="knowledge-status">

          <BookOpen size={17} />

          Evidence Knowledge Base

        </div>

      </div>


      {/* Search */}

      <div className="knowledge-search">

        <Search size={18} />

        <input
          type="text"
          placeholder="Search intervention, SDOH factor or evidence..."
          value={search}
          onChange={(event) =>
            setSearch(event.target.value)
          }
        />

      </div>


      {/* Main */}

      <div className="knowledge-layout">

        {/* Evidence List */}

        <div className="dashboard-card">

          <div className="card-header">

            <div>

              <h2>
                Evidence & Knowledge
              </h2>

              <p>
                Supporting information for
                intervention decisions.
              </p>

            </div>

            <ShieldCheck size={21} />

          </div>


          <div className="evidence-list">

            {filteredEvidence.length === 0 ? (

              <div className="empty-state">

                No matching evidence found.

              </div>

            ) : (

              filteredEvidence.map(
                (item) => (

                  <button
                    key={item.id}
                    className={`evidence-item ${
                      selected.id === item.id
                        ? "selected"
                        : ""
                    }`}
                    onClick={() =>
                      setSelected(item)
                    }
                  >

                    <div className="evidence-icon">

                      <BookOpen size={17} />

                    </div>


                    <div className="evidence-item-main">

                      <strong>
                        {item.intervention}
                      </strong>

                      <span>
                        {item.sdohFactor}
                      </span>

                      <div className="evidence-tags">

                        {item.tags
                          .slice(0, 2)
                          .map((tag) => (

                            <small key={tag}>
                              {tag}
                            </small>

                          ))}

                      </div>

                    </div>


                    <div className="evidence-confidence">

                      <strong>
                        {Math.round(
                          item.confidence * 100
                        )}
                        %
                      </strong>

                      <span>
                        Confidence
                      </span>

                    </div>

                  </button>

                )
              )

            )}

          </div>

        </div>


        {/* Details */}

        <div className="dashboard-card">

          <div className="card-header">

            <div>

              <h2>
                Evidence Explanation
              </h2>

              <p>
                Why this recommendation is
                supported.
              </p>

            </div>

            <Sparkles size={21} />

          </div>


          <div className="knowledge-title">

            <div className="knowledge-title-icon">

              <BookOpen size={21} />

            </div>

            <div>

              <h2>
                {selected.intervention}
              </h2>

              <span>
                {selected.sdohFactor}
              </span>

            </div>

          </div>


          {/* Confidence */}

          <div className="confidence-box">

            <div>

              <span>
                Evidence Confidence
              </span>

              <strong>
                {Math.round(
                  selected.confidence * 100
                )}
                %
              </strong>

            </div>

            <div className="confidence-bar">

              <div
                style={{
                  width:
                    `${selected.confidence * 100}%`,
                }}
              />

            </div>

            <small>
              Evidence Level:{" "}
              {selected.evidenceLevel}
            </small>

          </div>


          {/* Explanation */}

          <div className="knowledge-section">

            <div className="section-title">

              <Sparkles size={17} />

              <h3>
                Evidence-Based Explanation
              </h3>

            </div>

            <p>
              {selected.explanation}
            </p>

          </div>


          {/* Source */}

          <div className="knowledge-section">

            <div className="section-title">

              <ShieldCheck size={17} />

              <h3>
                Supporting Evidence
              </h3>

            </div>

            <p>
              {selected.source}
            </p>

          </div>


          {/* Source type */}

          <div className="source-card">

            <CheckCircle size={18} />

            <div>

              <strong>
                Knowledge Source
              </strong>

              <span>
                {selected.sourceType}
              </span>

            </div>

          </div>


          {/* Tags */}

          <div className="knowledge-tags">

            {selected.tags.map((tag) => (

              <span key={tag}>
                {tag}
              </span>

            ))}

          </div>

        </div>

      </div>

    </div>
  );
}

export default KnowledgeIntelligence;
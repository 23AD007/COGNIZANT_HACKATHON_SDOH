import { useParams, Link } from "react-router-dom";
import {
  ArrowLeft,
  MapPin,
  TrendingUp,
  Brain,
  ShieldCheck,
} from "lucide-react";

import { memberDetails } from "../data/mockData";
import RiskBadge from "../components/RiskBadge";

function MemberDetails() {
  const { memberId } = useParams();

  const member = memberDetails[memberId];

  if (!member) {
    return (
      <div className="empty-member">
        <h2>Member not found</h2>

        <Link to="/members">
          Return to Member Explorer
        </Link>
      </div>
    );
  }

  return (
    <div className="member-details-page">

      {/* Header */}

      <div className="member-detail-header">

        <div>

          <Link
            to="/members"
            className="back-link"
          >
            <ArrowLeft size={16} />
            Back to Members
          </Link>

          <div className="member-title">

            <div>
              <h1>
                Member {member.id}
              </h1>

              <p>
                Individual SDOH risk profile
              </p>
            </div>

            <RiskBadge
              level={member.riskLevel}
            />

          </div>

        </div>


        <div className="member-location">

          <MapPin size={17} />

          <span>{member.county}</span>

        </div>

      </div>


      {/* Risk Overview */}

      <div className="detail-grid">

        <div className="detail-card risk-overview">

          <div className="detail-card-header">

            <div>
              <h2>Risk Overview</h2>

              <p>
                Model-generated member risk
              </p>
            </div>

            <TrendingUp size={20} />

          </div>


          <div className="risk-score-large">

            <strong>
              {Math.round(
                member.riskScore * 100
              )}
              %
            </strong>

            <span>
              Overall Risk Score
            </span>

          </div>


          <RiskBadge
            level={member.riskLevel}
          />

        </div>


        {/* County Context */}

        <div className="detail-card">

          <div className="detail-card-header">

            <div>
              <h2>County Context</h2>

              <p>
                Geographic SDOH context
              </p>
            </div>

            <MapPin size={20} />

          </div>


          <div className="county-context">

            <div>
              <span>County</span>
              <strong>
                {member.county}
              </strong>
            </div>

            <div>
              <span>County Risk</span>
              <strong>
                {Math.round(
                  member.countyContext.countyRisk *
                    100
                )}
                %
              </strong>
            </div>

            <div>
              <span>County Rank</span>
              <strong>
                #{member.countyContext.countyRank}
              </strong>
            </div>

            <div>
              <span>Primary SDOH</span>
              <strong>
                {member.countyContext.primarySdoh}
              </strong>
            </div>

          </div>

        </div>

      </div>


      {/* SDOH Drivers */}

      <div className="detail-card">

        <div className="detail-card-header">

          <div>
            <h2>SDOH Drivers</h2>

            <p>
              Factors contributing to the
              member's risk profile
            </p>
          </div>

          <Brain size={20} />

        </div>


        <div className="sdoh-driver-list">

          {member.sdohDrivers.map(
            (driver) => (

              <div
                className="sdoh-driver"
                key={driver.name}
              >

                <div className="driver-name">
                  {driver.name}
                </div>

                <div className="driver-progress">

                  <div
                    style={{
                      width: `${driver.score}%`,
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


      {/* Interventions */}

      <div className="detail-card">

        <div className="detail-card-header">

          <div>
            <h2>
              Personalized Intervention
              Priorities
            </h2>

            <p>
              Ranked recommendations for this member
            </p>
          </div>

          <TrendingUp size={20} />

        </div>


        <div className="personalized-interventions">

          {member.interventions.map(
            (intervention) => (

              <div
                className="personalized-intervention"
                key={intervention.rank}
              >

                <div className="intervention-rank">
                  #{intervention.rank}
                </div>

                <div className="personalized-info">

                  <strong>
                    {intervention.name}
                  </strong>

                  <p>
                    {intervention.reason}
                  </p>

                </div>

                <div className="intervention-score">

                  {Math.round(
                    intervention.score * 100
                  )}
                  %

                </div>

              </div>

            )
          )}

        </div>

      </div>


      {/* Evidence */}

      <div className="evidence-card">

        <div className="evidence-header">

          <ShieldCheck size={22} />

          <div>
            <h2>
              Evidence-Based Explanation
            </h2>

            <span>
              {member.evidence.source}
            </span>
          </div>

        </div>


        <h3>
          {member.evidence.title}
        </h3>

        <p>
          {member.evidence.explanation}
        </p>

      </div>

    </div>
  );
}

export default MemberDetails;
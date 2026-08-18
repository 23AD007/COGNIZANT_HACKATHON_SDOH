function RiskExplanation({ member }) {
  return (
    <div className="detail-card">

      <div className="detail-card-header">
        <div>
          <h2>Explainable Risk Score</h2>
          <p>Factors contributing to member risk</p>
        </div>
      </div>

      <div className="risk-score-large">
        <strong>
          {(member.riskScore * 100).toFixed(0)}%
        </strong>

        <span>
          {member.riskLevel} Risk
        </span>
      </div>

      <div className="sdoh-driver-list">

        {member.sdohDrivers.map((driver) => (
          <div className="sdoh-driver" key={driver.name}>

            <span className="driver-name">
              {driver.name}
            </span>

            <div className="driver-progress">
              <div
                style={{
                  width: `${driver.score}%`
                }}
              />
            </div>

            <strong>
              {driver.score}%
            </strong>

          </div>
        ))}

      </div>

      <div className="evidence-card">
        <h3>Why is this member high risk?</h3>

        <p>
          Transportation is the strongest identified
          SDOH contributor, followed by economic stability
          and food access.
        </p>
      </div>

    </div>
  );
}

export default RiskExplanation;
function ExplainableRisk({ member }) {
  if (!member) return null;

  const drivers = member.sdohDrivers || [];

  return (
    <div className="detail-card">

      <div className="detail-card-header">
        <div>
          <h2>Explainable Risk Score</h2>
          <p>Why this member has this risk level</p>
        </div>
      </div>

      <div className="risk-score-large">
        <strong>
          {Math.round(member.riskScore * 100)}
        </strong>
        <span>Overall Risk Score</span>
      </div>

      <div className="sdoh-driver-list">

        {drivers.map((driver) => (
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

      {drivers.length > 0 && (
        <div className="evidence-mini">

          <strong>
            Primary Risk Driver
          </strong>

          <p>
            {drivers[0].name} is the strongest
            identified SDOH contributor with a
            score of {drivers[0].score}%.
          </p>

        </div>
      )}

    </div>
  );
}

export default ExplainableRisk;
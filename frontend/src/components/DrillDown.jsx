function DrillDown({ member, county, population }) {

  return (
    <div className="detail-card">

      <h2>Risk Context</h2>

      <div className="county-context">

        <div>
          <span>Member</span>
          <strong>{member?.id}</strong>
        </div>

        <div>
          <span>Member Risk</span>
          <strong>
            {member
              ? Math.round(member.riskScore * 100)
              : 0}%
          </strong>
        </div>

        <div>
          <span>County</span>
          <strong>{county?.name}</strong>
        </div>

        <div>
          <span>County Risk</span>
          <strong>
            {county
              ? Math.round(county.riskScore * 100)
              : 0}%
          </strong>
        </div>

      </div>

    </div>
  );
}

export default DrillDown;
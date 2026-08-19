function RiskBadge({ level }) {
  const className = `risk-badge ${level.toLowerCase()}`;

  return (
    <span className={className}>
      {level}
    </span>
  );
}

export default RiskBadge;
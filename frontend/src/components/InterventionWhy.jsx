function InterventionWhy({ intervention }) {
  if (!intervention) return null;

  return (
    <div className="detail-card">

      <h2>Why This Intervention?</h2>

      <p style={{ marginTop: "12px" }}>
        {intervention.reason}
      </p>

      <div className="evidence-mini">

        <strong>Evidence</strong>

        <p>
          {intervention.evidence ||
            "Supported by member-level and population-level SDOH indicators."}
        </p>

      </div>

    </div>
  );
}

export default InterventionWhy;
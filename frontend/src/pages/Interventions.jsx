import { Link } from "react-router-dom";
import { Building2, UserRound } from "lucide-react";

function Interventions() {
  return <div className="interventions-page"><div className="page-header"><div><h1>Recommendations</h1><p>Choose a real member or county to view the recommendations available for that selection.</p></div></div><div className="intervention-summary"><Link to="/members" className="summary-card"><UserRound size={24} /><div><span>Member flow</span><strong>View Member Recommendations</strong><p>Choose a member to see that member’s risk, SDOH, clinical factors, and recommendations.</p></div></Link><Link to="/county-map" className="summary-card"><Building2 size={24} /><div><span>County flow</span><strong>View County Priorities</strong><p>Choose a county to see its population risk, SDOH context, and population priorities.</p></div></Link></div><div className="dashboard-card"><div className="card-header"><div><h2>No global ranking is shown</h2><p>The API provides recommendations only in the context of a selected member or selected county.</p></div></div></div></div>;
}

export default Interventions;

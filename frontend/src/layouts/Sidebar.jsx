import {
  LayoutDashboard,
  Users,
  Map,
  GitBranch,
  Brain,
  MessageSquare,
} from "lucide-react";

function Sidebar() {
  return (
    <aside className="sidebar">

      <div className="sidebar-logo">
        <div className="logo-icon">S</div>

        <div>
          <h2>SDOH</h2>
          <span>Intelligence</span>
        </div>
      </div>

      <nav className="sidebar-nav">

        <a href="/">
          <LayoutDashboard size={20} />
          <span>Dashboard</span>
        </a>

        <a href="/members">
          <Users size={20} />
          <span>Members</span>
        </a>

        <a href="/county-map">
          <Map size={20} />
          <span>County Risk Map</span>
        </a>

        <a href="/interventions">
          <GitBranch size={20} />
          <span>Interventions</span>
        </a>

        <a href="/knowledge">
          <Brain size={20} />
          <span>Knowledge Intelligence</span>
        </a>

        <a href="/chat">
          <MessageSquare size={20} />
          <span>AI Assistant</span>
        </a>

      </nav>

    </aside>
  );
}

export default Sidebar;
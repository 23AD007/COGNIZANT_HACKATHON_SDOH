import { Bell, Search } from "lucide-react";

function Navbar() {
  return (
    <header className="navbar">

      <div className="navbar-title">
        <h1>SDOH Intelligence Platform</h1>
        <p>Social Determinants of Health Decision Support</p>
      </div>

      <div className="navbar-actions">

        <div className="search-box">
          <Search size={18} />
          <input
            type="text"
            placeholder="Search..."
          />
        </div>

        <button className="icon-button">
          <Bell size={20} />
        </button>

        <div className="profile">
          <div className="profile-avatar">
            A
          </div>

          <div>
            <strong>Administrator</strong>
            <span>SDOH Analyst</span>
          </div>
        </div>

      </div>

    </header>
  );
}

export default Navbar;
import { useMemo, useState } from "react";
import { Search, Filter, Eye } from "lucide-react";

import { members } from "../data/mockData";
import RiskBadge from "../components/RiskBadge";

function Members() {
  const [search, setSearch] = useState("");
  const [riskFilter, setRiskFilter] = useState("All");
  const [countyFilter, setCountyFilter] = useState("All");

  const counties = [
    "All",
    ...new Set(members.map((member) => member.county)),
  ];

  const filteredMembers = useMemo(() => {
    return members.filter((member) => {

      const matchesSearch =
        member.id
          .toLowerCase()
          .includes(search.toLowerCase());

      const matchesRisk =
        riskFilter === "All" ||
        member.riskLevel === riskFilter;

      const matchesCounty =
        countyFilter === "All" ||
        member.county === countyFilter;

      return (
        matchesSearch &&
        matchesRisk &&
        matchesCounty
      );
    });
  }, [search, riskFilter, countyFilter]);

  return (
    <div className="members-page">

      {/* Header */}

      <div className="page-header">

        <div>
          <h1>Member Explorer</h1>

          <p>
            Search and analyze individual member
            SDOH risk profiles.
          </p>
        </div>

      </div>


      {/* Filters */}

      <div className="filter-card">

        <div className="member-search">

          <Search size={18} />

          <input
            type="text"
            placeholder="Search Member ID..."
            value={search}
            onChange={(event) =>
              setSearch(event.target.value)
            }
          />

        </div>


        <div className="filter-group">

          <Filter size={17} />

          <select
            value={riskFilter}
            onChange={(event) =>
              setRiskFilter(event.target.value)
            }
          >

            <option value="All">
              All Risk Levels
            </option>

            <option value="High">
              High Risk
            </option>

            <option value="Medium">
              Medium Risk
            </option>

            <option value="Low">
              Low Risk
            </option>

          </select>

        </div>


        <div className="filter-group">

          <select
            value={countyFilter}
            onChange={(event) =>
              setCountyFilter(event.target.value)
            }
          >

            {counties.map((county) => (
              <option
                key={county}
                value={county}
              >
                {county}
              </option>
            ))}

          </select>

        </div>

      </div>


      {/* Results */}

      <div className="member-table-card">

        <div className="table-header">

          <div>
            <h2>Members</h2>

            <p>
              {filteredMembers.length} members found
            </p>
          </div>

        </div>


        <div className="table-wrapper">

          <table className="member-table">

            <thead>

              <tr>
                <th>Member ID</th>
                <th>Risk Level</th>
                <th>Risk Score</th>
                <th>Primary SDOH</th>
                <th>Secondary SDOH</th>
                <th>County</th>
                <th>Action</th>
              </tr>

            </thead>


            <tbody>

              {filteredMembers.map((member) => (

                <tr key={member.id}>

                  <td>
                    <strong>
                      {member.id}
                    </strong>
                  </td>

                  <td>
                    <RiskBadge
                      level={member.riskLevel}
                    />
                  </td>

                  <td>

                    <div className="risk-score-cell">

                      <span>
                        {Math.round(
                          member.riskScore * 100
                        )}
                        %
                      </span>

                      <div className="risk-progress">

                        <div
                          style={{
                            width: `${
                              member.riskScore * 100
                            }%`,
                          }}
                        />

                      </div>

                    </div>

                  </td>

                  <td>
                    {member.primarySdoh}
                  </td>

                  <td>
                    {member.secondarySdoh}
                  </td>

                  <td>
                    {member.county}
                  </td>

                  <td>

                    <a
                      href={`/members/${member.id}`}
                      className="view-member-button"
                    >

                      <Eye size={16} />

                      View

                    </a>

                  </td>

                </tr>

              ))}


              {filteredMembers.length === 0 && (

                <tr>

                  <td
                    colSpan="7"
                    className="empty-state"
                  >
                    No members found.
                  </td>

                </tr>

              )}

            </tbody>

          </table>

        </div>

      </div>

    </div>
  );
}

export default Members;
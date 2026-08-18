import {
  Users,
  AlertTriangle,
  Activity,
  MapPinned,
  TrendingUp,
} from "lucide-react";

import {
  dashboardStats,
  riskDistribution,
  sdohDrivers,
  topInterventions,
} from "../data/mockData";

import StatCard from "../components/StatCard";

import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";

function Dashboard() {
  return (
    <div className="dashboard">

      {/* Header */}

      <div className="page-header">

        <div>
          <h1>SDOH Overview</h1>

          <p>
            Population health and social determinants
            intelligence dashboard.
          </p>
        </div>

        <div className="dashboard-status">
          <span className="status-dot"></span>
          System Ready
        </div>

      </div>


      {/* Statistics */}

      <div className="stats-grid">

        <StatCard
          title="Total Members"
          value={dashboardStats.totalMembers.toLocaleString()}
          subtitle="Members analyzed"
          icon={<Users size={24} />}
        />

        <StatCard
          title="High Risk Members"
          value={dashboardStats.highRiskMembers.toLocaleString()}
          subtitle="Require priority attention"
          icon={<AlertTriangle size={24} />}
        />

        <StatCard
          title="Average Risk"
          value={`${Math.round(dashboardStats.averageRisk * 100)}%`}
          subtitle="Population risk score"
          icon={<Activity size={24} />}
        />

        <StatCard
          title="Counties"
          value={dashboardStats.totalCounties}
          subtitle="Geographic areas analyzed"
          icon={<MapPinned size={24} />}
        />

      </div>


      {/* Charts */}

      <div className="dashboard-grid">

        {/* Risk Distribution */}

        <div className="dashboard-card">

          <div className="card-header">
            <div>
              <h2>Risk Distribution</h2>
              <p>Member population by risk level</p>
            </div>
          </div>

          <div className="chart-container">

            <ResponsiveContainer width="100%" height="100%">

              <PieChart>

                <Pie
                  data={riskDistribution}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={100}
                  label
                >
                  {riskDistribution.map(
                    (entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                      />
                    )
                  )}
                </Pie>

                <Tooltip />

              </PieChart>

            </ResponsiveContainer>

          </div>

        </div>


        {/* SDOH Drivers */}

        <div className="dashboard-card">

          <div className="card-header">

            <div>
              <h2>Top SDOH Drivers</h2>
              <p>
                Population-level social risk factors
              </p>
            </div>

          </div>

          <div className="bar-chart">

            <ResponsiveContainer
              width="100%"
              height="100%"
            >

              <BarChart
                data={sdohDrivers}
                layout="vertical"
              >

                <CartesianGrid
                  strokeDasharray="3 3"
                />

                <XAxis
                  type="number"
                  domain={[0, 100]}
                />

                <YAxis
                  dataKey="name"
                  type="category"
                  width={130}
                />

                <Tooltip />

                <Bar
                  dataKey="score"
                  radius={[0, 6, 6, 0]}
                />

              </BarChart>

            </ResponsiveContainer>

          </div>

        </div>

      </div>


      {/* Intervention Section */}

      <div className="dashboard-card">

        <div className="card-header">

          <div>
            <h2>Priority Interventions</h2>

            <p>
              Highest-ranked interventions across
              the population
            </p>
          </div>

          <TrendingUp size={22} />

        </div>


        <div className="intervention-list">

          {topInterventions.map(
            (intervention) => (

              <div
                className="intervention-row"
                key={intervention.rank}
              >

                <div className="rank">
                  #{intervention.rank}
                </div>

                <div className="intervention-info">

                  <strong>
                    {intervention.name}
                  </strong>

                  <span>
                    {intervention.affectedMembers}
                    {" "}
                    potentially affected members
                  </span>

                </div>

                <div className="priority-score">

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

    </div>
  );
}

export default Dashboard;
import { BrowserRouter, Routes, Route } from "react-router-dom";

import MainLayout from "./layouts/MainLayout";

import Dashboard from "./pages/Dashboard";
import Members from "./pages/Members";
import MemberDetails from "./pages/MemberDetails";
import CountyRiskMap from "./pages/CountyRiskMap";
import CountyDetails from "./pages/CountyDetails";
import Interventions from "./pages/Interventions";
import KnowledgeIntelligence from "./pages/KnowledgeIntelligence";

function App() {
  return (
    <BrowserRouter>

      <MainLayout>

        <Routes>

          <Route path="/" element={<Dashboard />} />

          <Route
            path="/members"
            element={<Members />}
          />
          
          <Route
            path="/members/:memberId"
            element={<MemberDetails />}
          />

          <Route
            path="/county-map"
            element={<CountyRiskMap />}
          />

          <Route path="/counties/:countyFips" element={<CountyDetails />} />

          <Route
            path="/interventions"
            element={<Interventions />}
          />

          <Route
            path="/knowledge"
            element={<KnowledgeIntelligence />}
          />

        </Routes>

      </MainLayout>

    </BrowserRouter>
  );
}

export default App;

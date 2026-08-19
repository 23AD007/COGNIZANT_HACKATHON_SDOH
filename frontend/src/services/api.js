const BASE_URL = import.meta.env.VITE_API_URL || "/api";

async function request(path) {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const error = new Error(payload?.detail || `API request failed: ${response.status}`);
    error.status = response.status;
    error.detail = payload?.detail;
    throw error;
  }
  return response.json();
}

export const getCounties = () => request("/counties");
export const getCountyLocations = () => request("/counties/locations");
export const getCounty = (countyFips) => request(`/counties/${countyFips}`);
export const getCountyRecommendations = (countyFips) => request(`/counties/${countyFips}/recommendations`);

export const api = {
  health: () => request("/health"),
  dashboard: () => request("/dashboard/summary"),
  members: () => request("/members"),
  member: (memberId) => request(`/members/${memberId}`),
  sdoh: (memberId) => request(`/members/${memberId}/sdoh`),
  clinical: (memberId) => request(`/members/${memberId}/clinical`),
  risk: (memberId) => request(`/members/${memberId}/risk`),
  location: (memberId) => request(`/members/${memberId}/location`),
  recommendations: (memberId) => request(`/members/${memberId}/recommendations`),
  getCounties,
  getCountyLocations,
  getCounty,
  getCountyRecommendations,
  knowledge: () => request("/knowledge"),
};

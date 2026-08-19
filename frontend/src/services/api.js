const BASE_URL = import.meta.env.VITE_API_URL || "/api";

async function request(path) {
  const response = await fetch(`${BASE_URL}${path}`);
  if (!response.ok) {
    const error = new Error(`API request failed: ${response.status}`);
    error.status = response.status;
    throw error;
  }
  return response.json();
}

export const api = {
  dashboard: () => request("/dashboard/summary"),
  members: () => request("/members"),
  member: (memberId) => request(`/members/${memberId}`),
  sdoh: (memberId) => request(`/members/${memberId}/sdoh`),
  clinical: (memberId) => request(`/members/${memberId}/clinical`),
  risk: (memberId) => request(`/members/${memberId}/risk`),
  location: (memberId) => request(`/members/${memberId}/location`),
  recommendations: (memberId) => request(`/members/${memberId}/recommendations`),
};

export const dashboardStats = {
  totalMembers: 12450,
  highRiskMembers: 2184,
  mediumRiskMembers: 4962,
  lowRiskMembers: 5304,
  averageRisk: 0.64,
  totalCounties: 42,
};

export const riskDistribution = [
  {
    name: "High Risk",
    value: 2184,
  },
  {
    name: "Medium Risk",
    value: 4962,
  },
  {
    name: "Low Risk",
    value: 5304,
  },
];

export const sdohDrivers = [
  {
    name: "Transportation",
    score: 82,
  },
  {
    name: "Food Access",
    score: 74,
  },
  {
    name: "Economic Stability",
    score: 68,
  },
  {
    name: "Housing",
    score: 61,
  },
  {
    name: "Healthcare Access",
    score: 56,
  },
];

export const topInterventions = [
  {
    rank: 1,
    name: "Transportation Assistance",
    score: 0.91,
    affectedMembers: 842,
  },
  {
    rank: 2,
    name: "Food Assistance",
    score: 0.84,
    affectedMembers: 716,
  },
  {
    rank: 3,
    name: "Healthcare Access",
    score: 0.79,
    affectedMembers: 634,
  },
  {
    rank: 4,
    name: "Financial Assistance",
    score: 0.72,
    affectedMembers: 521,
  },
];

export const members = [
  {
    id: "M001",
    riskScore: 0.87,
    riskLevel: "High",
    primarySdoh: "Transportation",
    secondarySdoh: "Economic Stability",
    county: "County A",
  },
  {
    id: "M002",
    riskScore: 0.81,
    riskLevel: "High",
    primarySdoh: "Food Access",
    secondarySdoh: "Housing",
    county: "County B",
  },
  {
    id: "M003",
    riskScore: 0.63,
    riskLevel: "Medium",
    primarySdoh: "Healthcare Access",
    secondarySdoh: "Transportation",
    county: "County A",
  },
  {
    id: "M004",
    riskScore: 0.42,
    riskLevel: "Low",
    primarySdoh: "Housing",
    secondarySdoh: "Food Access",
    county: "County C",
  },
];

export const memberDetails = {
  M001: {
    id: "M001",
    riskScore: 0.87,
    riskLevel: "High",
    county: "County A",

    sdohDrivers: [
      {
        name: "Transportation",
        score: 82,
      },
      {
        name: "Economic Stability",
        score: 68,
      },
      {
        name: "Food Access",
        score: 55,
      },
      {
        name: "Healthcare Access",
        score: 48,
      },
    ],

    countyContext: {
      countyRisk: 0.72,
      countyRank: 8,
      primarySdoh: "Transportation",
    },

    interventions: [
      {
        rank: 1,
        name: "Transportation Assistance",
        score: 0.91,
        reason:
          "High transportation-related SDOH risk combined with member-level vulnerability.",
      },
      {
        rank: 2,
        name: "Healthcare Access",
        score: 0.79,
        reason:
          "Healthcare access barriers may increase difficulty in obtaining appropriate care.",
      },
      {
        rank: 3,
        name: "Financial Assistance",
        score: 0.72,
        reason:
          "Economic stability is an important contributing SDOH factor.",
      },
    ],

    evidence: {
      title: "Transportation-related risk",
      explanation:
        "Transportation is the strongest identified SDOH driver for this member. The intervention ranking prioritizes transportation assistance because the member's transportation risk is high and the county also shows elevated transportation-related SDOH risk.",
      source:
        "Knowledge Intelligence / Evidence Base",
    },
  },

  M002: {
    id: "M002",
    riskScore: 0.81,
    riskLevel: "High",
    county: "County B",

    sdohDrivers: [
      {
        name: "Food Access",
        score: 79,
      },
      {
        name: "Housing",
        score: 71,
      },
      {
        name: "Economic Stability",
        score: 63,
      },
    ],

    countyContext: {
      countyRisk: 0.69,
      countyRank: 12,
      primarySdoh: "Food Access",
    },

    interventions: [
      {
        rank: 1,
        name: "Food Assistance",
        score: 0.89,
        reason:
          "Food access is the strongest identified SDOH driver.",
      },
      {
        rank: 2,
        name: "Housing Assistance",
        score: 0.76,
        reason:
          "Housing instability contributes significantly to the member's SDOH risk.",
      },
      {
        rank: 3,
        name: "Financial Assistance",
        score: 0.68,
        reason:
          "Economic stability is another contributing factor.",
      },
    ],

    evidence: {
      title: "Food access risk",
      explanation:
        "Food access represents the strongest SDOH driver for this member and is therefore prioritized in the intervention ranking.",
      source:
        "Knowledge Intelligence / Evidence Base",
    },
  },
};

export const counties = [
  {
    id: "C001",
    name: "County A",
    state: "State A",
    riskScore: 0.78,
    riskLevel: "High",
    members: 842,
    primarySdoh: "Transportation",

    sdohDrivers: [
      {
        name: "Transportation",
        score: 82,
      },
      {
        name: "Food Access",
        score: 71,
      },
      {
        name: "Housing",
        score: 64,
      },
      {
        name: "Economic Stability",
        score: 59,
      },
    ],

    intervention: "Transportation Assistance",

    latitude: 40.7128,
    longitude: -74.0060,
  },

  {
    id: "C002",
    name: "County B",
    state: "State A",
    riskScore: 0.64,
    riskLevel: "Medium",
    members: 615,
    primarySdoh: "Food Access",

    sdohDrivers: [
      {
        name: "Food Access",
        score: 76,
      },
      {
        name: "Housing",
        score: 62,
      },
      {
        name: "Healthcare Access",
        score: 54,
      },
    ],

    intervention: "Food Assistance",

    latitude: 40.7306,
    longitude: -73.9352,
  },

  {
    id: "C003",
    name: "County C",
    state: "State A",
    riskScore: 0.39,
    riskLevel: "Low",
    members: 421,
    primarySdoh: "Housing",

    sdohDrivers: [
      {
        name: "Housing",
        score: 48,
      },
      {
        name: "Food Access",
        score: 42,
      },
      {
        name: "Transportation",
        score: 37,
      },
    ],

    intervention: "Housing Assistance",

    latitude: 40.6782,
    longitude: -73.9442,
  },

  {
    id: "C004",
    name: "County D",
    state: "State A",
    riskScore: 0.86,
    riskLevel: "High",
    members: 1032,
    primarySdoh: "Economic Stability",

    sdohDrivers: [
      {
        name: "Economic Stability",
        score: 88,
      },
      {
        name: "Housing",
        score: 74,
      },
      {
        name: "Healthcare Access",
        score: 68,
      },
    ],

    intervention: "Financial Assistance",

    latitude: 40.6501,
    longitude: -73.9496,
  },
];

export const interventionRanking = [
  {
    rank: 1,
    intervention: "Transportation Assistance",
    category: "Transportation",
    score: 0.91,
    priority: "Critical",
    affectedMembers: 842,
    target: "High-risk members",
    reason:
      "Transportation is a dominant SDOH risk factor and may create barriers to accessing healthcare and essential services.",
    evidence:
      "Supported by member-level transportation risk and geographic SDOH context.",
  },
  {
    rank: 2,
    intervention: "Food Assistance",
    category: "Food Access",
    score: 0.84,
    priority: "High",
    affectedMembers: 716,
    target: "Food-insecure members",
    reason:
      "Food access represents a significant contributing SDOH factor.",
    evidence:
      "Supported by food-access indicators and population-level geographic risk.",
  },
  {
    rank: 3,
    intervention: "Healthcare Access Support",
    category: "Healthcare Access",
    score: 0.79,
    priority: "High",
    affectedMembers: 634,
    target: "Members with access barriers",
    reason:
      "Healthcare access barriers can reduce the ability to obtain appropriate care.",
    evidence:
      "Supported by healthcare-access-related SDOH indicators.",
  },
  {
    rank: 4,
    intervention: "Financial Assistance",
    category: "Economic Stability",
    score: 0.72,
    priority: "Medium",
    affectedMembers: 521,
    target: "Economically vulnerable members",
    reason:
      "Economic instability can contribute to multiple downstream social and healthcare barriers.",
    evidence:
      "Supported by economic stability indicators.",
  },
  {
    rank: 5,
    intervention: "Housing Assistance",
    category: "Housing",
    score: 0.68,
    priority: "Medium",
    affectedMembers: 438,
    target: "Housing-vulnerable members",
    reason:
      "Housing instability may contribute to persistent social and health-related risk.",
    evidence:
      "Supported by housing-related SDOH indicators.",
  },
];

export const knowledgeEvidence = [
  {
    id: "E001",
    intervention: "Transportation Assistance",
    sdohFactor: "Transportation",
    evidenceLevel: "High",
    confidence: 0.94,
    sourceType: "SDOH Knowledge Base",
    source:
      "Transportation barriers can reduce access to healthcare, preventive services, employment, and essential resources.",
    explanation:
      "Transportation is identified as a major SDOH risk factor. The intervention is prioritized because the member or population shows elevated transportation-related risk.",
    tags: [
      "Transportation",
      "Healthcare Access",
      "High Risk",
    ],
  },

  {
    id: "E002",
    intervention: "Food Assistance",
    sdohFactor: "Food Access",
    evidenceLevel: "High",
    confidence: 0.91,
    sourceType: "SDOH Knowledge Base",
    source:
      "Limited access to nutritious and affordable food can contribute to poor health outcomes and increased social vulnerability.",
    explanation:
      "Food access is a significant SDOH driver for the selected population, making food assistance an appropriate intervention.",
    tags: [
      "Food Access",
      "Nutrition",
      "High Risk",
    ],
  },

  {
    id: "E003",
    intervention: "Healthcare Access Support",
    sdohFactor: "Healthcare Access",
    evidenceLevel: "Moderate",
    confidence: 0.86,
    sourceType: "SDOH Knowledge Base",
    source:
      "Barriers to healthcare access may prevent individuals from obtaining timely and appropriate healthcare services.",
    explanation:
      "Healthcare access indicators suggest that additional support may help reduce barriers to care.",
    tags: [
      "Healthcare Access",
      "Care Navigation",
    ],
  },

  {
    id: "E004",
    intervention: "Financial Assistance",
    sdohFactor: "Economic Stability",
    evidenceLevel: "Moderate",
    confidence: 0.81,
    sourceType: "SDOH Knowledge Base",
    source:
      "Economic instability can affect the ability to obtain healthcare, food, housing, transportation, and other essential resources.",
    explanation:
      "Elevated economic instability makes financial assistance a potentially valuable intervention.",
    tags: [
      "Economic Stability",
      "Financial Risk",
    ],
  },

  {
    id: "E005",
    intervention: "Housing Assistance",
    sdohFactor: "Housing",
    evidenceLevel: "Moderate",
    confidence: 0.79,
    sourceType: "SDOH Knowledge Base",
    source:
      "Housing instability can create persistent barriers to health, safety, and access to essential services.",
    explanation:
      "Housing-related SDOH indicators support consideration of housing assistance.",
    tags: [
      "Housing",
      "Housing Stability",
    ],
  },
];
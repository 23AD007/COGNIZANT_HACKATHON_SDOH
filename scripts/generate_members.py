import os
import random
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
from faker import Faker
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not found in .env")

NUM_MEMBERS = 10_000
SEED = 42

random.seed(SEED)
np.random.seed(SEED)

fake = Faker()
Faker.seed(SEED)

engine = create_engine(DATABASE_URL)


# ============================================================
# DATABASE SETUP
# ============================================================

def create_database_objects():

    with engine.begin() as conn:

        # Enable PostGIS
        conn.execute(text("""
            CREATE EXTENSION IF NOT EXISTS postgis;
        """))

        # ----------------------------------------------------
        # Geography table
        # ----------------------------------------------------

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS geographies (
                id SERIAL PRIMARY KEY,
                geography_code VARCHAR(20) UNIQUE NOT NULL,
                geography_name VARCHAR(100) NOT NULL,
                state VARCHAR(50),
                county VARCHAR(100),
                median_income NUMERIC(12,2),
                poverty_rate NUMERIC(5,2),
                food_insecurity_rate NUMERIC(5,2),
                housing_instability_rate NUMERIC(5,2),
                transportation_barrier_rate NUMERIC(5,2),
                healthcare_access_score NUMERIC(5,2),
                geom GEOMETRY(POLYGON, 4326)
            );
        """))

        # ----------------------------------------------------
        # Member table
        # ----------------------------------------------------

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS members (
                id SERIAL PRIMARY KEY,

                member_id VARCHAR(50) UNIQUE NOT NULL,

                age INTEGER NOT NULL,
                gender VARCHAR(20),

                annual_income NUMERIC(12,2),
                employment_status VARCHAR(50),

                chronic_condition_count INTEGER,
                primary_conditions TEXT,

                ed_visits INTEGER,
                hospitalizations INTEGER,
                primary_care_visits INTEGER,

                medication_adherence NUMERIC(5,2),

                transportation_access NUMERIC(5,2),
                housing_stability NUMERIC(5,2),
                food_access NUMERIC(5,2),
                healthcare_access NUMERIC(5,2),

                geography_code VARCHAR(20),

                latitude DOUBLE PRECISION,
                longitude DOUBLE PRECISION,

                geom GEOMETRY(POINT, 4326),

                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                CONSTRAINT fk_member_geography
                    FOREIGN KEY (geography_code)
                    REFERENCES geographies(geography_code)
            );
        """))

        # Indexes
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_members_geography
            ON members(geography_code);
        """))

        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_members_geom
            ON members USING GIST(geom);
        """))

        print("Database tables created.")


# ============================================================
# SYNTHETIC GEOGRAPHIES
# ============================================================

def generate_geographies():

    states = [
        "Tamil Nadu",
        "Karnataka",
        "Kerala",
        "Telangana",
        "Andhra Pradesh"
    ]

    geographies = []

    for i in range(1, 51):

        state = random.choice(states)

        # Base socioeconomic conditions
        median_income = np.random.normal(55000, 15000)
        median_income = max(18000, min(median_income, 120000))

        poverty_rate = np.random.normal(15, 7)
        poverty_rate = max(3, min(poverty_rate, 45))

        food_insecurity = (
            5
            + poverty_rate * 0.45
            + np.random.normal(0, 3)
        )

        food_insecurity = max(3, min(food_insecurity, 45))

        housing_instability = (
            5
            + poverty_rate * 0.35
            + np.random.normal(0, 4)
        )

        housing_instability = max(
            2,
            min(housing_instability, 40)
        )

        transportation_barrier = (
            10
            + poverty_rate * 0.5
            + np.random.normal(0, 5)
        )

        transportation_barrier = max(
            5,
            min(transportation_barrier, 60)
        )

        healthcare_access = (
            100
            - poverty_rate * 0.8
            - transportation_barrier * 0.25
            + np.random.normal(0, 5)
        )

        healthcare_access = max(
            20,
            min(healthcare_access, 100)
        )

        geography_code = f"GEO{i:04d}"

        # Synthetic coordinates
        latitude = np.random.uniform(8.0, 17.5)
        longitude = np.random.uniform(74.0, 82.0)

        # Create a small synthetic polygon
        size = 0.08

        polygon = (
            f"POLYGON(("
            f"{longitude-size} {latitude-size},"
            f"{longitude+size} {latitude-size},"
            f"{longitude+size} {latitude+size},"
            f"{longitude-size} {latitude+size},"
            f"{longitude-size} {latitude-size}"
            f"))"
        )

        geographies.append({
            "geography_code": geography_code,
            "geography_name": f"Synthetic Geography {i}",
            "state": state,
            "county": f"Synthetic County {i}",
            "median_income": round(median_income, 2),
            "poverty_rate": round(poverty_rate, 2),
            "food_insecurity_rate": round(food_insecurity, 2),
            "housing_instability_rate": round(housing_instability, 2),
            "transportation_barrier_rate": round(
                transportation_barrier, 2
            ),
            "healthcare_access_score": round(
                healthcare_access, 2
            ),
            "latitude": latitude,
            "longitude": longitude,
            "polygon": polygon
        })

    return geographies


# ============================================================
# INSERT GEOGRAPHIES
# ============================================================

def insert_geographies(geographies):

    with engine.begin() as conn:

        # Clear old synthetic data
        conn.execute(
            text("DELETE FROM members")
        )

        conn.execute(
            text("DELETE FROM geographies")
        )

        for geo in geographies:

            conn.execute(text("""
                INSERT INTO geographies (
                    geography_code,
                    geography_name,
                    state,
                    county,
                    median_income,
                    poverty_rate,
                    food_insecurity_rate,
                    housing_instability_rate,
                    transportation_barrier_rate,
                    healthcare_access_score,
                    geom
                )
                VALUES (
                    :code,
                    :name,
                    :state,
                    :county,
                    :income,
                    :poverty,
                    :food,
                    :housing,
                    :transportation,
                    :healthcare,
                    ST_GeomFromText(:polygon, 4326)
                )
            """), {
                "code": geo["geography_code"],
                "name": geo["geography_name"],
                "state": geo["state"],
                "county": geo["county"],
                "income": geo["median_income"],
                "poverty": geo["poverty_rate"],
                "food": geo["food_insecurity_rate"],
                "housing": geo["housing_instability_rate"],
                "transportation": geo[
                    "transportation_barrier_rate"
                ],
                "healthcare": geo[
                    "healthcare_access_score"
                ],
                "polygon": geo["polygon"]
            })

    print(f"Inserted {len(geographies)} synthetic geographies.")


# ============================================================
# MEMBER GENERATION
# ============================================================

def generate_member(geo):

    # --------------------------------------------------------
    # Demographics
    # --------------------------------------------------------

    age = int(
        np.clip(
            np.random.normal(48, 20),
            18,
            90
        )
    )

    gender = random.choice([
        "Male",
        "Female",
        "Other"
    ])

    # --------------------------------------------------------
    # Income
    # --------------------------------------------------------

    income = np.random.normal(
        geo["median_income"],
        geo["median_income"] * 0.25
    )

    income = max(10000, min(income, 200000))

    # --------------------------------------------------------
    # Employment
    # --------------------------------------------------------

    if age > 65:
        employment = "Retired"

    else:

        employment_probability = (
            0.75
            - geo["poverty_rate"] / 100
        )

        if random.random() < employment_probability:
            employment = "Employed"
        else:
            employment = random.choice([
                "Unemployed",
                "Self-employed"
            ])

    # --------------------------------------------------------
    # Chronic conditions
    # --------------------------------------------------------

    age_factor = max(0, (age - 30) / 60)

    condition_lambda = (
        0.5
        + age_factor * 2.5
    )

    chronic_count = int(
        np.random.poisson(condition_lambda)
    )

    chronic_count = min(chronic_count, 6)

    conditions = [
        "Hypertension",
        "Diabetes",
        "Asthma",
        "COPD",
        "Heart Disease",
        "Arthritis"
    ]

    if chronic_count > 0:

        primary_conditions = random.sample(
            conditions,
            min(chronic_count, len(conditions))
        )

        primary_conditions = ", ".join(
            primary_conditions
        )

    else:
        primary_conditions = None

    # --------------------------------------------------------
    # SDOH
    # --------------------------------------------------------

    # Lower score = worse condition

    transportation_access = 100 - np.random.normal(
        geo["transportation_barrier_rate"],
        8
    )

    transportation_access = np.clip(
        transportation_access,
        0,
        100
    )

    housing_stability = 100 - np.random.normal(
        geo["housing_instability_rate"],
        8
    )

    housing_stability = np.clip(
        housing_stability,
        0,
        100
    )

    food_access = 100 - np.random.normal(
        geo["food_insecurity_rate"],
        8
    )

    food_access = np.clip(
        food_access,
        0,
        100
    )

    healthcare_access = np.random.normal(
        geo["healthcare_access_score"],
        7
    )

    healthcare_access = np.clip(
        healthcare_access,
        0,
        100
    )

    # --------------------------------------------------------
    # Medication adherence
    # --------------------------------------------------------

    adherence = (
        70
        + income / 10000
        + healthcare_access * 0.1
        + np.random.normal(0, 10)
    )

    adherence -= chronic_count * 2

    adherence = np.clip(
        adherence,
        0,
        100
    )

    # --------------------------------------------------------
    # Healthcare utilization
    # --------------------------------------------------------

    base_ed = (
        0.5
        + chronic_count * 0.7
    )

    # Poor transportation → more ED visits
    base_ed += (
        (100 - transportation_access)
        / 30
    )

    # Poor housing → more utilization
    base_ed += (
        (100 - housing_stability)
        / 40
    )

    ed_visits = int(
        np.random.poisson(
            max(base_ed, 0.1)
        )
    )

    # Hospitalization probability
    hospitalization_rate = (
        0.05
        + chronic_count * 0.05
        + ed_visits * 0.02
    )

    hospitalizations = int(
        np.random.poisson(
            max(hospitalization_rate, 0.01)
        )
    )

    primary_care_visits = int(
        np.random.poisson(
            2
            + chronic_count * 1.5
        )
    )

    # --------------------------------------------------------
    # Location
    # --------------------------------------------------------

    latitude = (
        geo["latitude"]
        + np.random.normal(0, 0.025)
    )

    longitude = (
        geo["longitude"]
        + np.random.normal(0, 0.025)
    )

    return {
        "member_id": f"MEM-{uuid.uuid4().hex[:10].upper()}",
        "age": age,
        "gender": gender,
        "annual_income": round(income, 2),
        "employment_status": employment,
        "chronic_condition_count": chronic_count,
        "primary_conditions": primary_conditions,
        "ed_visits": ed_visits,
        "hospitalizations": hospitalizations,
        "primary_care_visits": primary_care_visits,
        "medication_adherence": round(adherence, 2),
        "transportation_access": round(
            transportation_access,
            2
        ),
        "housing_stability": round(
            housing_stability,
            2
        ),
        "food_access": round(
            food_access,
            2
        ),
        "healthcare_access": round(
            healthcare_access,
            2
        ),
        "geography_code": geo["geography_code"],
        "latitude": latitude,
        "longitude": longitude
    }


# ============================================================
# GENERATE MEMBERS
# ============================================================

def generate_members(geographies):

    members = []

    for _ in range(NUM_MEMBERS):

        geo = random.choice(geographies)

        member = generate_member(geo)

        members.append(member)

    return members


# ============================================================
# INSERT MEMBERS
# ============================================================

def insert_members(members):

    with engine.begin() as conn:

        for member in members:

            conn.execute(text("""
                INSERT INTO members (
                    member_id,
                    age,
                    gender,
                    annual_income,
                    employment_status,
                    chronic_condition_count,
                    primary_conditions,
                    ed_visits,
                    hospitalizations,
                    primary_care_visits,
                    medication_adherence,
                    transportation_access,
                    housing_stability,
                    food_access,
                    healthcare_access,
                    geography_code,
                    latitude,
                    longitude,
                    geom
                )
                VALUES (
                    :member_id,
                    :age,
                    :gender,
                    :income,
                    :employment,
                    :conditions,
                    :primary_conditions,
                    :ed_visits,
                    :hospitalizations,
                    :primary_care_visits,
                    :adherence,
                    :transportation,
                    :housing,
                    :food,
                    :healthcare,
                    :geography,
                    :latitude,
                    :longitude,
                    ST_SetSRID(
                        ST_MakePoint(
                            :longitude,
                            :latitude
                        ),
                        4326
                    )
                )
            """), {
                "member_id": member["member_id"],
                "age": member["age"],
                "gender": member["gender"],
                "income": member["annual_income"],
                "employment": member["employment_status"],
                "conditions": member[
                    "chronic_condition_count"
                ],
                "primary_conditions": member[
                    "primary_conditions"
                ],
                "ed_visits": member["ed_visits"],
                "hospitalizations": member[
                    "hospitalizations"
                ],
                "primary_care_visits": member[
                    "primary_care_visits"
                ],
                "adherence": member[
                    "medication_adherence"
                ],
                "transportation": member[
                    "transportation_access"
                ],
                "housing": member[
                    "housing_stability"
                ],
                "food": member["food_access"],
                "healthcare": member[
                    "healthcare_access"
                ],
                "geography": member[
                    "geography_code"
                ],
                "latitude": member["latitude"],
                "longitude": member["longitude"]
            })

    print(f"Inserted {len(members)} synthetic members.")


# ============================================================
# VALIDATE
# ============================================================

def validate_database():

    with engine.connect() as conn:

        result = conn.execute(text("""
            SELECT COUNT(*)
            FROM members
        """))

        member_count = result.scalar()

        result = conn.execute(text("""
            SELECT COUNT(*)
            FROM geographies
        """))

        geography_count = result.scalar()

        print()
        print("==============================")
        print("DATABASE VALIDATION")
        print("==============================")
        print(f"Members     : {member_count}")
        print(f"Geographies : {geography_count}")

        result = conn.execute(text("""
            SELECT
                member_id,
                age,
                chronic_condition_count,
                ed_visits,
                transportation_access,
                housing_stability,
                food_access,
                geography_code
            FROM members
            LIMIT 5
        """))

        rows = result.fetchall()

        print()
        print("Sample members:")

        for row in rows:
            print(row)


# ============================================================
# MAIN
# ============================================================

def main():

    print("Starting synthetic member generation...")

    create_database_objects()

    print("Generating synthetic geographies...")

    geographies = generate_geographies()

    insert_geographies(geographies)

    print("Generating synthetic members...")

    members = generate_members(geographies)

    print("Inserting members into PostgreSQL...")

    insert_members(members)

    validate_database()

    print()
    print("===================================")
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("===================================")


if __name__ == "__main__":
    main()
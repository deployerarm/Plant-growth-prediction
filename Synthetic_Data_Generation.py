import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

os.makedirs("Datasets", exist_ok=True)

plant_classes = [
    'Arive-Dantu', 'Basale', 'Betel', 'Crape_Jasmine', 'Curry', 'Mint', 'Neem',
    'Oleander', 'Parijata', 'Peepal', 'Pomegranate', 'Rasna', 'Rose_apple',
    'Roxburgh_fig', 'Sandalwood', 'Tulsi'
]

# ---------------------------------------------
# REAL-WORLD GROWTH PROFILES PER SPECIES
# ---------------------------------------------
GROWTH_RATES = {
    'Mint': 0.35, 'Tulsi': 0.30, 'Basale': 0.32,
    'Curry': 0.22, 'Betel': 0.28, 'Neem': 0.18,
    'Sandalwood': 0.10, 'Peepal': 0.15, 'Rose_apple': 0.20,
    'Oleander': 0.25, 'Roxburgh_fig': 0.12,
    'Crape_Jasmine': 0.24, 'Parijata': 0.26,
    'Pomegranate': 0.23, 'Arive-Dantu': 0.27, 'Rasna': 0.22
}

# ---------------------------------------------
# Generate dataset
# ---------------------------------------------
def generate_realistic_dataset(samples=15000):
    rows = []
    start_date = datetime(2023, 1, 1)

    samples_per_plant = samples // len(plant_classes)

    for plant in plant_classes:

        # starting height varies per species (seedling size)
        height = np.random.uniform(3, 12)
        health = np.random.uniform(0.6, 0.9)

        # initial environment
        temp = np.random.uniform(23, 32)
        humidity = np.random.uniform(45, 85)
        soil = np.random.uniform(30, 70)
        ph = np.random.uniform(5.8, 7.2)
        light = np.random.uniform(12000, 22000)

        for i in range(samples_per_plant):

            date = start_date + timedelta(days=i)

            # -----------------------------
            # Smooth environment transitions
            # -----------------------------
            temp += np.random.normal(0, 0.4)
            humidity += np.random.normal(0, 1.2)
            soil += np.random.normal(0, 1.0)
            ph += np.random.normal(0, 0.05)
            light += np.random.normal(0, 500)

            # clamp environment
            temp = np.clip(temp, 20, 36)
            humidity = np.clip(humidity, 35, 95)
            soil = np.clip(soil, 20, 85)
            ph = np.clip(ph, 5.0, 7.5)
            light = np.clip(light, 8000, 26000)

            # -----------------------------
            # Realistic growth dynamics
            # -----------------------------
            base_growth = GROWTH_RATES[plant]

            # environmental impact
            env_factor = (
                -0.02 * abs(temp - 28)
                -0.015 * abs(humidity - 60)
                -0.008 * abs(soil - 50)
                -0.05 * abs(ph - 6.5)
                - (abs(light - 18000) / 50000)
            )

            daily_growth = base_growth + env_factor
            daily_growth = max(0.01, daily_growth)  # plants never shrink

            height += daily_growth

            # -----------------------------
            # Leaf area (depends on species + height)
            # -----------------------------
            leaf_area = (height ** 1.2) * np.random.uniform(1.8, 2.4)

            # -----------------------------
            # Health score (smooth changes)
            # -----------------------------
            health_change = (
                +0.001 * (soil - 50)
                -0.002 * abs(temp - 28)
                -0.001 * abs(humidity - 60)
                -0.003 * abs(ph - 6.5)
                + np.random.normal(0, 0.01)
            )
            health += health_change
            health = np.clip(health, 0.1, 1.0)

            # -----------------------------
            # Plant age
            # -----------------------------
            age_days = i + np.random.randint(5, 20)

            rows.append([
                plant,
                date.strftime("%Y-%m-%d"),
                age_days,
                temp,
                humidity,
                soil,
                ph,
                light,
                height,
                leaf_area,
                health
            ])

    df = pd.DataFrame(rows, columns=[
        "plant_species", "date", "plant_age_days",
        "temperature", "humidity", "soil_moisture", "soil_ph", "light_intensity",
        "height_cm", "leaf_area_cm2", "health_score"
    ])

    df.to_csv("Datasets/plant_growth_dataset.csv", index=False)
    print("Dataset successfully saved: Datasets/plant_growth_dataset.csv")


generate_realistic_dataset()

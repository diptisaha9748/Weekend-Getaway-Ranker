import pandas as pd

#df = pd.read_csv("Top Indian Places to Visit.csv")
#print(df.head())
#print(df.columns)

from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import MinMaxScaler

# --------------------------------------------------
# City coordinates (approximate)
# --------------------------------------------------
city_coordinates = {
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bangalore": (12.9716, 77.5946),
    "Chennai": (13.0827, 80.2707),
    "Kolkata": (22.5726, 88.3639),
    "Hyderabad": (17.3850, 78.4867),
    "Jaipur": (26.9124, 75.7873),
    "Agra": (27.1767, 78.0081),
    "Goa": (15.2993, 74.1240)
}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    return R * c

def add_distance_column(df, source_city):
    src_lat, src_lon = city_coordinates[source_city]

    df["distance_km"] = df["City"].apply(
        lambda city: haversine(
            src_lat,
            src_lon,
            city_coordinates[city][0],
            city_coordinates[city][1]
        ) if city in city_coordinates else None
    )

    return df

def recommend_weekend_getaways(source_city, top_n=5):
    df = pd.read_csv("Top Indian Places to Visit.csv")

    df = add_distance_column(df, source_city)

    weekend_df = df[df["distance_km"] <= 350].copy()

    scaler = MinMaxScaler()
    weekend_df[["dist_norm", "rating_norm", "popularity_norm"]] = scaler.fit_transform(
        weekend_df[
            ["distance_km", "Google review rating", "Number of google review in lakhs"]
        ]
    )

    weekend_df["dist_norm"] = 1 - weekend_df["dist_norm"]

    weekend_df["final_score"] = (
        0.4 * weekend_df["dist_norm"]
        + 0.35 * weekend_df["rating_norm"]
        + 0.25 * weekend_df["popularity_norm"]
    )

    return weekend_df.sort_values("final_score", ascending=False).head(top_n)

if __name__ == "__main__":
    for city in ["Delhi", "Mumbai", "Bangalore"]:
        print(f"\nTop Weekend Getaways from {city}")
        print(
            recommend_weekend_getaways(city)[
                [
                    "Name",
                    "City",
                    "State",
                    "distance_km",
                    "Google review rating",
                    "Number of google review in lakhs",
                    "final_score",
                ]
            ]
        )
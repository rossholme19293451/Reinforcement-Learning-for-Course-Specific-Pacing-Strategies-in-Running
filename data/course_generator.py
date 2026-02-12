import pandas as pd
import numpy as np

def generate_flat_course_csv(total_distance,  filename, step=1, elevation=0.0):

    distance = np.arange(0, total_distance + step, step)

    elevation_array = np.full_like(distance, elevation, dtype=float)

    df = pd.DataFrame({
        "Distance": distance,
        "Elevation": elevation_array
    })

    df.to_csv(f"../data/elevation_profiles/{filename}", index=False)
    print(f"Wrote to {filename}")
    return df

if __name__ == "__main__":
    df = generate_flat_course_csv(total_distance=1000, filename="flat_1km.csv")
    print(df.head())
import pandas as pd
import numpy as np

def generate_flat_course_csv(total_distance,  filename, step=1, elevation=0.0):
    """
    Generate a flat course csv file for testing and validation
    """

    #create a distance array from 0 to total_distance in step increments
    distance = np.arange(0, total_distance + step, step)

    #create constant elevation array with length total_distance
    elevation_array = np.full_like(distance, elevation, dtype=float)

    #combine into data frame
    df = pd.DataFrame({
        "Distance": distance,
        "Elevation": elevation_array
    })

    #save to csv file
    df.to_csv(f"../data/elevation_profiles/{filename}", index=False)
    print(f"Wrote to {filename}")
    return df

if __name__ == "__main__":
    df = generate_flat_course_csv(total_distance=10000, filename="flat_10km.csv")
    print(df.head())
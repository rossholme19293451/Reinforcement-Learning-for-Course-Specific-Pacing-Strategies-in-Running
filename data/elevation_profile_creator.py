import csv

import gpxpy
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

#takes gpx file, returns array of distance and elevation at that point
def parse_GPX_to_points(gpx):
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            total_distance = 0
            prev = None
            for point in segment.points:
                if prev is not None:
                    dist = point.distance_2d(prev)
                    total_distance += dist

                points.append((total_distance, point.elevation))
                prev = point

    points = np.array(points, dtype=float)
    dists, idx = np.unique(points[:, 0], return_index=True)
    elevs = points[idx, 1]
    return dists, elevs

#interpolates 1m steps between the distance and elevation points
def resample_to_1m(dists, elevs, step=0.5):
    interp_func = interp1d(dists, elevs, kind='linear', fill_value='extrapolate')
    new_dists = np.arange(0, dists[-1], step=step)
    new_elevs = interp_func(new_dists)
    new_elevs = savgol_filter(new_elevs, window_length=201, polyorder=4)
    return new_dists, new_elevs

#save the elevation profile to csv file
def save_csv(filename, dists, elevs):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Distance', 'Elevation'])
        writer.writerows(zip(dists, elevs))
        print(f"Saved to {filename}")
        f.close()

if __name__ == "__main__":
    # load gpx file
    with open("course_GPXs/Ryde_10.gpx", "r") as f:
        gpx = gpxpy.parse(f)

    dists, elevs = parse_GPX_to_points(gpx)
    new_dists, new_elevs = resample_to_1m(dists, elevs)
    filename = "../data/elevation_profiles/Ryde_10.csv"
    save_csv(filename, new_dists, new_elevs)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(50, 4))
    plt.plot(new_dists, new_elevs, color='blue', label='Interpolated Elevation (1m)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Race Elevation Profile (1m resolution)')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(50, 4))
    plt.plot(dists, elevs, 'o', color='red', label='Original GPX Points')
    plt.plot(new_dists, new_elevs, '-', color='blue', label='Interpolated (1m)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Original vs. Interpolated Elevation')
    plt.legend()
    plt.grid(True)
    plt.show()
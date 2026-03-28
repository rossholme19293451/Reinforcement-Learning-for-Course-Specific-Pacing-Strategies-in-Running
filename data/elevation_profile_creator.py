import csv
import gpxpy
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

def parse_GPX_to_points(gpx):
    """
    Extract cumulative distance and elevation from a GPX file.
    """
    points = []
    for track in gpx.tracks:
        for segment in track.segments:
            total_distance = 0
            prev = None
            for point in segment.points:
                #calculate 2D distance between consecutive GPS points
                if prev is not None:
                    dist = point.distance_2d(prev)
                    total_distance += dist

                points.append((total_distance, point.elevation))
                prev = point

    #remove duplicate distances
    points = np.array(points, dtype=float)
    dists, idx = np.unique(points[:, 0], return_index=True)
    elevs = points[idx, 1]
    return dists, elevs

#interpolates 1m steps between the distance and elevation points
def resample_to_1m(dists, elevs, step=1):
    """
    Resample irregular GPS data to uniform 1m spacing and smooth noise.
    """
    #linear interpolation to 1m resolution
    interp_func = interp1d(dists, elevs, kind='linear', fill_value='extrapolate')
    new_dists = np.arange(0, dists[-1], step=step)
    new_elevs = interp_func(new_dists)

    #Savitzky-Golay filter to remove GPS noise, 201m window, poly order of 4
    new_elevs = savgol_filter(new_elevs, window_length=201, polyorder=4)
    return new_dists, new_elevs

def save_csv(filename, dists, elevs):
    """
    Save elevation profile to CSV file.
    """
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Distance', 'Elevation'])
        writer.writerows(zip(dists, elevs))
        print(f"Saved to {filename}")
        f.close()

if __name__ == "__main__":
    # load gpx file
    with open("course_GPXs/Oxford HM.gpx", "r") as f:
        gpx = gpxpy.parse(f)

    #process: parse -> resample to 1m -> smooth
    dists, elevs = parse_GPX_to_points(gpx)
    new_dists, new_elevs = resample_to_1m(dists, elevs)
    filename = "elevation_profiles/Oxford_HM.csv"
    save_csv(filename, new_dists, new_elevs)

    #visualise smoothed elevation profile
    plt.figure(figsize=(50, 4))
    plt.plot(new_dists, new_elevs, color='blue', label='Interpolated Elevation (1m)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Race Elevation Profile (1m resolution)')
    plt.legend()
    plt.grid(True)
    plt.show()

    #compare original GPS points with processed profile
    plt.figure(figsize=(50, 4))
    plt.plot(dists, elevs, 'o', color='red', label='Original GPX Points')
    plt.plot(new_dists, new_elevs, '-', color='blue', label='Interpolated (1m)')
    plt.xlabel('Distance (m)')
    plt.ylabel('Elevation (m)')
    plt.title('Original vs. Interpolated Elevation')
    plt.legend()
    plt.grid(True)
    plt.show()
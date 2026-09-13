"""Preserve centers below nearby physical camera heights; no ground is inferred."""
import numpy as np


def below_camera_centers(means, camera_centers, station_ids, up, nearest=3):
    means = np.asarray(means, np.float64)
    camera_centers = np.asarray(camera_centers, np.float64)
    up = np.asarray(up, np.float64)
    if means.ndim != 2 or means.shape[1] != 3 or not np.isfinite(means).all():
        raise ValueError('Finite Gaussian centers required')
    if camera_centers.shape != (len(station_ids), 3) or not np.isfinite(camera_centers).all():
        raise ValueError('Finite camera centers and physical station labels required')
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) == 0:
        raise ValueError('Validated up direction required')
    if type(nearest) is not int or nearest < 2:
        raise ValueError('At least two nearest physical stations required')
    up = up / np.linalg.norm(up)
    labels = list(map(str, station_ids))
    unique = sorted(set(labels))
    if len(unique) < nearest:
        raise ValueError('Insufficient distinct physical stations')
    # Exact duplicate faces do not reweight the reference or local heights.
    centers = np.stack([np.unique(camera_centers[np.array(labels)==key],axis=0).mean(0) for key in unique])
    radius = np.linalg.norm(centers - centers.mean(0), axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError('Degenerate camera baseline')
    for station, center in zip(unique, centers):
        members = camera_centers[np.array(labels) == station]
        if np.linalg.norm(members - center, axis=1).max() > radius * 1e-8:
            raise ValueError('Faces of one physical station disagree in position')
    distances = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    if distances.min() <= radius * 1e-8:
        raise ValueError('Co-located capture IDs need physical-station grouping')
    origin = centers[0]
    camera_height = (centers - origin) @ up
    protect = np.zeros(len(means), bool)
    heights = np.empty(len(means))
    centers_height = (means - origin) @ up
    for first in range(0, len(means), 4096):
        last = min(first + 4096, len(means))
        delta = means[first:last, None, :] - centers[None, :, :]
        horizontal = delta - (delta @ up)[:, :, None] * up
        squared = np.einsum('nsi,nsi->ns', horizontal, horizontal)
        cutoff = np.partition(squared, nearest - 1, axis=1)[:, nearest - 1]
        tie = 128 * np.finfo(np.float64).eps * (squared.max(1) + radius**2)
        included = squared <= (cutoff + tie)[:, None]
        height = np.where(included, camera_height, -np.inf).max(1)
        tolerance = 128 * np.finfo(np.float64).eps * (np.abs(centers_height[first:last]) + np.abs(height) + radius)
        protect[first:last] = centers_height[first:last] <= height + tolerance
        heights[first:last] = height
    return protect, dict(local_height=heights,center_height=centers_height,
                         physical_stations=len(unique),radius=radius,up=up,origin=origin)

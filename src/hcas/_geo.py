"""Geographic distance utilities."""

import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized Haversine great-circle distance in kilometres.

    Parameters
    ----------
    lat1, lon1, lat2, lon2 : float or array-like
        Coordinates in decimal degrees (broadcast-compatible).

    Returns
    -------
    distances : float or ndarray
        Great-circle distances in km.
    """
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) ** 2 + (
        np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2) ** 2
    )
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

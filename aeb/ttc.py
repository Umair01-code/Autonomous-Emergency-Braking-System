def calculate_ttc(distance, ego_speed, target_speed):
    """
    Calculate Time-to-Collision (TTC).

    distance: distance to target in meters
    ego_speed: ego vehicle speed in m/s
    target_speed: target vehicle speed in m/s
    """

    relative_speed = ego_speed - target_speed

    if distance <= 0:
        return 0.0

    if relative_speed <= 0:
        return float("inf")

    return distance / relative_speed

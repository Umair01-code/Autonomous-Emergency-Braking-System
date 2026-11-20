from ttc import calculate_ttc
from risk import assess_risk

import matplotlib.pyplot as plt


# -----------------------------
# Simulation Settings
# -----------------------------

ego_speed = 20.0
target_speed = 10.0

distance = 50.0

time_step = 0.1
simulation_time = 10.0


# -----------------------------
# AEB Settings
# -----------------------------

BRAKING_DECELERATION = 8.0

aeb_active = False


# -----------------------------
# Data Storage
# -----------------------------

times = []
distances = []
ego_speeds = []
ttcs = []
aeb_states = []


# -----------------------------
# Simulation
# -----------------------------

current_time = 0.0

print("\n===== AUTONOMOUS EMERGENCY BRAKING SIMULATOR =====\n")


while current_time <= simulation_time:

    # Calculate TTC
    ttc = calculate_ttc(
        distance,
        ego_speed,
        target_speed
    )

    # Calculate risk
    risk = assess_risk(ttc)

    # Activate AEB
    if risk == "AEB":
        aeb_active = True

    # Store data BEFORE updating vehicle state
    times.append(current_time)
    distances.append(distance)
    ego_speeds.append(ego_speed)

    if ttc == float("inf"):
        ttcs.append(None)
    else:
        ttcs.append(ttc)

    aeb_states.append(1 if aeb_active else 0)

    # Apply braking
    if aeb_active:
        ego_speed -= BRAKING_DECELERATION * time_step

        if ego_speed < 0:
            ego_speed = 0

    # Relative speed
    relative_speed = ego_speed - target_speed

    # Update distance
    distance -= relative_speed * time_step

    if distance < 0:
        distance = 0

    # Display
    if ttc == float("inf"):
        ttc_display = "INF"
    else:
        ttc_display = f"{ttc:.2f}"

    print(
        f"Time: {current_time:5.1f}s | "
        f"Distance: {distance:6.2f}m | "
        f"Ego Speed: {ego_speed:5.2f}m/s | "
        f"TTC: {ttc_display:>5} | "
        f"Risk: {risk:7} | "
        f"AEB: {'ON' if aeb_active else 'OFF'}"
    )

    # Collision detection
    if distance <= 0:
        print("\n🚨 COLLISION DETECTED!")
        break

    current_time += time_step


print("\n===== SIMULATION COMPLETE =====")


# -----------------------------
# Plot 1: Distance
# -----------------------------

plt.figure()

plt.plot(times, distances)

plt.xlabel("Time (seconds)")
plt.ylabel("Distance (meters)")
plt.title("Distance Between Ego and Target Vehicle")

plt.grid(True)

plt.show()


# -----------------------------
# Plot 2: Ego Speed
# -----------------------------

plt.figure()

plt.plot(times, ego_speeds)

plt.xlabel("Time (seconds)")
plt.ylabel("Ego Speed (m/s)")
plt.title("Ego Vehicle Speed")

plt.grid(True)

plt.show()


# -----------------------------
# Plot 3: TTC
# -----------------------------

plt.figure()

plt.plot(times, ttcs)

plt.xlabel("Time (seconds)")
plt.ylabel("TTC (seconds)")
plt.title("Time-to-Collision")

plt.grid(True)

plt.show()


# -----------------------------
# Plot 4: AEB Activation
# -----------------------------

plt.figure()

plt.step(times, aeb_states, where="post")

plt.xlabel("Time (seconds)")
plt.ylabel("AEB Status")
plt.title("AEB Activation")

plt.yticks([0, 1], ["OFF", "ON"])

plt.grid(True)

plt.show()
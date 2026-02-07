import os
import csv
import ac
import acsys
from third_party import sim_info
from collections import deque

# Settings
log_file = os.path.join(os.path.dirname(__file__), "third_party", "log.csv")

# Lap Count Tracking
lapcount = 0
current_lap_data = []

# REAL-TIME LABEL TRACKING (per session)
session_labels = {'Neutral': 0, 'Understeer': 0, 'Oversteer': 0}
current_lap_labels = {'Neutral': 0, 'Understeer': 0, 'Oversteer': 0}

# IMPROVED LABELING - Store recent history for rate of change calculations
history_size = 5  # Look at last 5 frames (~0.08 seconds at 60fps)
yaw_history = deque(maxlen=history_size)
lat_accel_history = deque(maxlen=history_size)
steer_history = deque(maxlen=history_size)
speed_history = deque(maxlen=history_size)

# Ensure the CSV file exists with header including Label
if not os.path.exists(log_file):
    with open(log_file, "w", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([
            "Lap", "CarModel", "Track", "TrackPos", "CurrentTime",
            "YawRate", "LateralAccel", "LongitudinalAccel", "VerticalAccel",
            "SteerAngle", "Speed", "LocalVelX", "LocalVelY", "LocalVelZ",
            "WheelSlipFL", "WheelSlipFR", "WheelSlipRL", "WheelSlipRR",
            "Throttle", "Brake", "Gear",
            "SurfaceGrip", "RoadTemp", "AirTemp",
            "Heading", "Pitch", "Roll",
            "CarX", "CarY", "CarZ",
            "SlipDiff", "YawGradient", "Label"  # Added YawGradient for better analysis
        ])

def acMain(ac_version):
    global l_lapcount, l_status, l_yaw, l_lataccel, l_slip_diff
    global l_conditions, l_speed, l_session_stats, l_lap_stats, l_recommendation
    
    appWindow = ac.newApp("ML Training Logger")
    ac.setSize(appWindow, 320, 340)
    
    ac.log("Improved ML Labeling Logger Initialized")
    ac.console("Using proper Bergman definition for labeling!")
    
    # === TITLE ===
    title = ac.addLabel(appWindow, "🤖 ML Training Logger v2")
    ac.setPosition(title, 3, 30)
    ac.setFontSize(title, 14)
    
    # === LAP COUNT ===
    l_lapcount = ac.addLabel(appWindow, "Laps: 0")
    ac.setPosition(l_lapcount, 3, 55)
    ac.setFontSize(l_lapcount, 12)
    
    # === STATUS ===
    l_status = ac.addLabel(appWindow, "Status: Waiting for Lap 1...")
    ac.setPosition(l_status, 3, 75)
    
    # === CURRENT BEHAVIOR ===
    l_slip_diff = ac.addLabel(appWindow, "Current: Neutral")
    ac.setPosition(l_slip_diff, 3, 95)
    ac.setFontSize(l_slip_diff, 12)
    
    # === CURRENT LAP DISTRIBUTION ===
    l_lap_stats = ac.addLabel(appWindow, "This Lap: N:0 U:0 O:0")
    ac.setPosition(l_lap_stats, 3, 115)
    
    # === SESSION TOTAL DISTRIBUTION ===
    l_session_stats = ac.addLabel(appWindow, "Session: N:0 U:0 O:0")
    ac.setPosition(l_session_stats, 3, 135)
    ac.setFontSize(l_session_stats, 11)
    
    # === DIVIDER ===
    divider = ac.addLabel(appWindow, "─" * 35)
    ac.setPosition(divider, 3, 155)
    
    # === TELEMETRY ===
    l_yaw = ac.addLabel(appWindow, "Yaw: 0.000 rad/s")
    ac.setPosition(l_yaw, 3, 170)
    
    l_lataccel = ac.addLabel(appWindow, "Lat Accel: 0.00 g")
    ac.setPosition(l_lataccel, 3, 190)
    
    l_speed = ac.addLabel(appWindow, "Speed: 0 km/h")
    ac.setPosition(l_speed, 3, 210)
    
    l_conditions = ac.addLabel(appWindow, "Grip: 1.00 | Road: 0°C")
    ac.setPosition(l_conditions, 3, 230)
    
    # === INFO ===
    info_label = ac.addLabel(appWindow, "Using proper Bergman definition")
    ac.setPosition(info_label, 3, 250)
    ac.setFontSize(info_label, 9)
    
    # === DIVIDER ===
    divider2 = ac.addLabel(appWindow, "─" * 35)
    ac.setPosition(divider2, 3, 265)
    
    # === RECOMMENDATION ===
    l_recommendation = ac.addLabel(appWindow, "💡 Drive to collect data")
    ac.setPosition(l_recommendation, 3, 280)
    ac.setFontSize(l_recommendation, 10)
    
    return "ML Training Logger"


def calculate_label_improved(yaw_rate, lat_accel, steer_angle, speed, slip_diff):
    """
    PROPER implementation of Bergman's (1966) definition
    """
    
    # STRICTER minimum speed threshold - low speed handling is unreliable
    if speed < 50:  # Below 50 km/h (~30 mph), can't judge handling reliably
        return "Neutral", 0, 0.0
    
    # Need enough history to calculate rate of change
    if len(yaw_history) < history_size or len(lat_accel_history) < history_size:
        return "Neutral", 0, 0.0
    
    # Calculate rate of change
    yaw_change = yaw_history[-1] - yaw_history[0]
    lat_accel_change = lat_accel_history[-1] - lat_accel_history[0]
    
    # === FILTER OUT LOSS OF CONTROL (SPINNING/CRASHING) ===
    speed_change = speed_history[-1] - speed_history[0] if len(speed_history) >= 2 else 0
    
    if abs(yaw_rate) > 1.0:  # Spinning out
        return "Neutral", 0, 0.0
    if speed_change < -15:  # Crash/major deceleration
        return "Neutral", 0, 0.0
    
    # Avoid division by zero
    if abs(lat_accel_change) < 0.01:
        if abs(lat_accel) < 0.3:
            return "Neutral", 0, 0.0
        if slip_diff > 0.15:
            return "Oversteer", 2, 0.0
        elif slip_diff < -0.15:
            return "Understeer", 1, 0.0
        else:
            return "Neutral", 0, 0.0
    
    yaw_gradient = yaw_change / lat_accel_change
    
    # Calculate expected yaw - BUT scale by speed (more lenient at medium speeds)
    steer_magnitude = abs(steer_angle)
    
    # Speed factor: less strict at medium speeds
    if speed < 80:  # 50-80 km/h: very lenient
        speed_factor = 0.3
    elif speed < 120:  # 80-120 km/h: moderate
        speed_factor = 0.5
    else:  # >120 km/h: full strictness
        speed_factor = 0.7
        
    expected_yaw_magnitude = steer_magnitude * (speed / 100.0) * speed_factor
    actual_yaw_magnitude = abs(yaw_rate)
    
    if expected_yaw_magnitude > 0.05:
        yaw_response_ratio = actual_yaw_magnitude / expected_yaw_magnitude
        
        # STRICTER thresholds to reduce false positives
        if yaw_response_ratio > 1.5 or yaw_gradient > 1.0:  # Was 1.3/0.8
            if slip_diff > 0.10:  # Was 0.08
                return "Oversteer", 2, yaw_gradient
        
        elif yaw_response_ratio < 0.6 or yaw_gradient < 0.15:  # Was 0.7/0.2
            if slip_diff < -0.10:  # Was -0.08
                return "Understeer", 1, yaw_gradient
    
    # Active cornering check - only at higher speeds
    if abs(lat_accel) > 0.5 and speed > 80:
        if slip_diff > 0.25:  # Was 0.20
            return "Oversteer", 2, yaw_gradient
        elif slip_diff < -0.25:  # Was -0.20
            return "Understeer", 1, yaw_gradient
    
    return "Neutral", 0, yaw_gradient


def get_recommendation():
    """Generate real-time recommendation based on current distribution"""
    total = sum(session_labels.values())
    
    if total < 100:
        return "💡 Keep driving to collect data"
    
    neutral_pct = (session_labels['Neutral'] / total) * 100
    understeer_pct = (session_labels['Understeer'] / total) * 100
    oversteer_pct = (session_labels['Oversteer'] / total) * 100
    
    if neutral_pct > 70:
        if understeer_pct < oversteer_pct:
            return "🔵 BRAKE LATE for understeer!"
        else:
            return "🔴 TRAIL BRAKE for oversteer!"
    elif understeer_pct < 10:
        return "🔵 Need understeer: brake late!"
    elif oversteer_pct < 10:
        return "🔴 Need oversteer: trail brake!"
    else:
        return "✅ Good balance - keep varying!"


def acUpdate(deltaT):
    global l_lapcount, l_status, l_yaw, l_lataccel, l_slip_diff
    global l_conditions, l_speed, l_session_stats, l_lap_stats, l_recommendation
    global lapcount, current_lap_data, session_labels, current_lap_labels
    global yaw_history, lat_accel_history, steer_history, speed_history
    
    # Fetch lap information
    laps = ac.getCarState(0, acsys.CS.LapCount)
    
    # Only process data from Lap 1 onwards (skip out-lap)
    if laps > 0:
        # === FETCH ALL TELEMETRY ===
        yaw_rate = sim_info.info.physics.localAngularVel[2]
        lateral_accel = sim_info.info.physics.accG[0]
        longitudinal_accel = sim_info.info.physics.accG[1]
        vertical_accel = sim_info.info.physics.accG[2]
        steer_angle = sim_info.info.physics.steerAngle
        speed = sim_info.info.physics.speedKmh
        local_vel_x = sim_info.info.physics.localVelocity[0]
        local_vel_y = sim_info.info.physics.localVelocity[1]
        local_vel_z = sim_info.info.physics.localVelocity[2]
        
        # Wheel slip
        wheel_slip_fl = sim_info.info.physics.wheelSlip[0]
        wheel_slip_fr = sim_info.info.physics.wheelSlip[1]
        wheel_slip_rl = sim_info.info.physics.wheelSlip[2]
        wheel_slip_rr = sim_info.info.physics.wheelSlip[3]
        
        # Driver inputs
        throttle = sim_info.info.physics.gas * 100
        brake = sim_info.info.physics.brake * 100
        gear = sim_info.info.physics.gear
        
        # Track conditions
        surface_grip = sim_info.info.graphics.surfaceGrip
        road_temp = sim_info.info.physics.roadTemp
        air_temp = sim_info.info.physics.airTemp
        
        # Position and orientation
        track_pos = sim_info.info.graphics.normalizedCarPosition
        current_time = sim_info.info.graphics.iCurrentTime
        heading = sim_info.info.physics.heading
        pitch = sim_info.info.physics.pitch
        roll = sim_info.info.physics.roll
        
        # Car coordinates
        car_coordinates = sim_info.info.graphics.carCoordinates
        car_x, car_y, car_z = car_coordinates[0], car_coordinates[1], car_coordinates[2]
        
        # === UPDATE HISTORY BUFFERS ===
        yaw_history.append(yaw_rate)
        lat_accel_history.append(lateral_accel)
        steer_history.append(steer_angle)
        speed_history.append(speed)
        
        # === CALCULATE SLIP DIFFERENTIAL ===
        front_slip_avg = (wheel_slip_fl + wheel_slip_fr) / 2.0
        rear_slip_avg = (wheel_slip_rl + wheel_slip_rr) / 2.0
        slip_diff = rear_slip_avg - front_slip_avg
        
        # === APPLY IMPROVED BERGMAN-BASED LABELING ===
        label_str, label_int, yaw_gradient = calculate_label_improved(
            yaw_rate, lateral_accel, steer_angle, speed, slip_diff
        )
        
        # Update label counters
        session_labels[label_str] += 1
        current_lap_labels[label_str] += 1
        
        # === DETECT LAP COMPLETION ===
        if laps > lapcount:
            # Get car info before resetting
            car_model = sim_info.info.static.carModel
            track_name = sim_info.info.static.track
            
            # Save previous lap data
            save_lap_data(lapcount, current_lap_data, car_model, track_name)
            
            # Update lap count
            lapcount = laps
            
            # Reset current lap data and history
            current_lap_data = []
            current_lap_labels = {'Neutral': 0, 'Understeer': 0, 'Oversteer': 0}
            yaw_history.clear()
            lat_accel_history.clear()
            steer_history.clear()
            speed_history.clear()
            
            ac.setText(l_status, "Lap {} Complete!".format(lapcount - 1))
        
        # === STORE DATA WITH LABEL ===
        current_lap_data.append([
            track_pos, current_time,
            yaw_rate, lateral_accel, longitudinal_accel, vertical_accel,
            steer_angle, speed, local_vel_x, local_vel_y, local_vel_z,
            wheel_slip_fl, wheel_slip_fr, wheel_slip_rl, wheel_slip_rr,
            throttle, brake, gear,
            surface_grip, road_temp, air_temp,
            heading, pitch, roll,
            car_x, car_y, car_z,
            slip_diff, yaw_gradient, label_str  # Include yaw_gradient for analysis
        ])
        
        # === UPDATE UI ===
        ac.setText(l_lapcount, "Laps: {}".format(lapcount))
        
        if lapcount > 0:
            ac.setText(l_status, "Logging Lap {}".format(lapcount))
        
        # Current behavior with color coding
        if label_str == "Oversteer":
            behavior_text = "Current: 🔴 OVERSTEER"
        elif label_str == "Understeer":
            behavior_text = "Current: 🔵 UNDERSTEER"
        else:
            behavior_text = "Current: 🟢 Neutral"
        ac.setText(l_slip_diff, behavior_text)
        
        # Current lap distribution
        lap_text = "This Lap: N:{} U:{} O:{}".format(
            current_lap_labels['Neutral'],
            current_lap_labels['Understeer'],
            current_lap_labels['Oversteer']
        )
        ac.setText(l_lap_stats, lap_text)
        
        # Session total distribution
        session_text = "Session: N:{} U:{} O:{}".format(
            session_labels['Neutral'],
            session_labels['Understeer'],
            session_labels['Oversteer']
        )
        ac.setText(l_session_stats, session_text)
        
        # Telemetry
        ac.setText(l_yaw, "Yaw: {:.3f} rad/s".format(yaw_rate))
        ac.setText(l_lataccel, "Lat Accel: {:.2f} g".format(lateral_accel))
        ac.setText(l_speed, "Speed: {:.0f} km/h".format(speed))
        ac.setText(l_conditions, "Grip: {:.2f} | Road: {:.0f}°C".format(surface_grip, road_temp))
        
        # Real-time recommendation
        recommendation = get_recommendation()
        ac.setText(l_recommendation, recommendation)
        
    else:
        # Still on out-lap
        ac.setText(l_status, "Out-Lap (Not Recording)")
        ac.setText(l_lapcount, "Laps: 0 (Out-Lap)")


def save_lap_data(lap_num, data, car_model, track_name):
    """Saves lap data with automatic labels to CSV file"""
    if not data or lap_num == 0:
        return
    
    with open(log_file, "a", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        
        for entry in data:
            writer.writerow([
                lap_num, car_model, track_name,
                entry[0], entry[1],  # TrackPos, CurrentTime
                entry[2], entry[3], entry[4], entry[5],  # Yaw, Accels
                entry[6], entry[7], entry[8], entry[9], entry[10],  # Steer, Speed, Velocity
                entry[11], entry[12], entry[13], entry[14],  # Wheel Slip
                entry[15], entry[16], entry[17],  # Throttle, Brake, Gear
                entry[18], entry[19], entry[20],  # Conditions
                entry[21], entry[22], entry[23],  # Heading, Pitch, Roll
                entry[24], entry[25], entry[26],  # Car Position
                entry[27], entry[28], entry[29]  # SlipDiff, YawGradient, Label
            ])
    
    ac.log("Saved {} labeled data points for Lap {}".format(len(data), lap_num))


def acShutdown():
    """Saves remaining lap data and prints final statistics"""
    if lapcount > 0 and current_lap_data:
        car_model = sim_info.info.static.carModel
        track_name = sim_info.info.static.track
        save_lap_data(lapcount, current_lap_data, car_model, track_name)
    
    # Log final session statistics
    total = sum(session_labels.values())
    if total > 0:
        ac.log("=== SESSION STATISTICS ===")
        ac.log("Total data points: {}".format(total))
        ac.log("Neutral: {} ({:.1f}%)".format(
            session_labels['Neutral'], 
            (session_labels['Neutral']/total)*100
        ))
        ac.log("Understeer: {} ({:.1f}%)".format(
            session_labels['Understeer'], 
            (session_labels['Understeer']/total)*100
        ))
        ac.log("Oversteer: {} ({:.1f}%)".format(
            session_labels['Oversteer'], 
            (session_labels['Oversteer']/total)*100
        ))
    
    ac.log("Improved ML Training Logger shutdown complete")
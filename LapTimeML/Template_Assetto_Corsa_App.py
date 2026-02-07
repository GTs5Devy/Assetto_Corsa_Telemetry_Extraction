import sys
import os
from third_party import sim_info
import ac
import acsys

# Labels for telemetry data display
l_lapcount = 0
l_status = 0
l_current_label = 0
l_lap_stats = 0
l_session_stats = 0
l_yaw = 0
l_lataccel = 0
l_slip_diff = 0
l_conditions = 0
l_speed = 0
l_recommendation = 0

lapcount = 0

# REAL-TIME LABEL TRACKING
session_labels = {'Neutral': 0, 'Understeer': 0, 'Oversteer': 0}
current_lap_labels = {'Neutral': 0, 'Understeer': 0, 'Oversteer': 0}

# LABELING THRESHOLDS (Based on Bergman 1966)
UNDERSTEER_THRESHOLD = -0.05
OVERSTEER_THRESHOLD = 0.05


def acMain(ac_version):
    global l_lapcount, l_status, l_current_label, l_lap_stats, l_session_stats
    global l_yaw, l_lataccel, l_slip_diff, l_conditions, l_speed, l_recommendation
    
    appWindow = ac.newApp("ML Training Monitor")
    ac.setSize(appWindow, 320, 320)
    
    ac.log("Real-Time ML Monitor UI Initialized")
    ac.console("ML Training Monitor Active - Real-time labeling enabled")
    
    # === TITLE ===
    title = ac.addLabel(appWindow, "🤖 ML Training Monitor")
    ac.setPosition(title, 3, 30)
    ac.setFontSize(title, 14)
    
    # === LAP COUNT ===
    l_lapcount = ac.addLabel(appWindow, "Laps: 0")
    ac.setPosition(l_lapcount, 3, 55)
    ac.setFontSize(l_lapcount, 12)
    
    # === STATUS ===
    l_status = ac.addLabel(appWindow, "Status: Waiting for Lap 1...")
    ac.setPosition(l_status, 3, 75)
    
    # === CURRENT BEHAVIOR LABEL ===
    l_current_label = ac.addLabel(appWindow, "Current: 🟢 Neutral")
    ac.setPosition(l_current_label, 3, 95)
    ac.setFontSize(l_current_label, 12)
    
    # === SLIP DIFFERENTIAL ===
    l_slip_diff = ac.addLabel(appWindow, "Slip Diff: 0.00")
    ac.setPosition(l_slip_diff, 3, 115)
    
    # === CURRENT LAP STATS ===
    l_lap_stats = ac.addLabel(appWindow, "This Lap: N:0 U:0 O:0")
    ac.setPosition(l_lap_stats, 3, 135)
    
    # === SESSION TOTAL STATS ===
    l_session_stats = ac.addLabel(appWindow, "Session: N:0 U:0 O:0")
    ac.setPosition(l_session_stats, 3, 155)
    ac.setFontSize(l_session_stats, 11)
    
    # === DIVIDER ===
    divider = ac.addLabel(appWindow, "─" * 35)
    ac.setPosition(divider, 3, 175)
    
    # === TELEMETRY ===
    l_yaw = ac.addLabel(appWindow, "Yaw: 0.000 rad/s")
    ac.setPosition(l_yaw, 3, 190)
    
    l_lataccel = ac.addLabel(appWindow, "Lat Accel: 0.00 g")
    ac.setPosition(l_lataccel, 3, 210)
    
    l_speed = ac.addLabel(appWindow, "Speed: 0 km/h")
    ac.setPosition(l_speed, 3, 230)
    
    l_conditions = ac.addLabel(appWindow, "Grip: 1.00 | Road: 0°C")
    ac.setPosition(l_conditions, 3, 250)
    
    # === DIVIDER ===
    divider2 = ac.addLabel(appWindow, "─" * 35)
    ac.setPosition(divider2, 3, 270)
    
    # === REAL-TIME RECOMMENDATION ===
    l_recommendation = ac.addLabel(appWindow, "💡 Drive smooth for neutral data")
    ac.setPosition(l_recommendation, 3, 285)
    ac.setFontSize(l_recommendation, 10)
    
    return "ML Training Monitor"


def calculate_label(slip_diff):
    """
    Apply Bergman's (1966) objective definition to label behavior
    Returns: tuple (label_string, label_int)
    """
    if slip_diff > OVERSTEER_THRESHOLD:
        return "Oversteer", 2
    elif slip_diff < UNDERSTEER_THRESHOLD:
        return "Understeer", 1
    else:
        return "Neutral", 0


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
    global l_lapcount, l_status, l_current_label, l_lap_stats, l_session_stats
    global l_yaw, l_lataccel, l_slip_diff, l_conditions, l_speed, l_recommendation
    global lapcount, session_labels, current_lap_labels
    
    # Fetch lap information
    laps = ac.getCarState(0, acsys.CS.LapCount)
    
    # Only process data from Lap 1 onwards
    if laps > 0:
        # === FETCH CRITICAL TELEMETRY ===
        yaw_rate = sim_info.info.physics.localAngularVel[2]
        lateral_accel = sim_info.info.physics.accG[0]
        speed = sim_info.info.physics.speedKmh
        
        # Wheel slip
        wheel_slip_fl = sim_info.info.physics.wheelSlip[0]
        wheel_slip_fr = sim_info.info.physics.wheelSlip[1]
        wheel_slip_rl = sim_info.info.physics.wheelSlip[2]
        wheel_slip_rr = sim_info.info.physics.wheelSlip[3]
        
        # Track conditions
        surface_grip = sim_info.info.graphics.surfaceGrip
        road_temp = sim_info.info.physics.roadTemp
        
        # === CALCULATE SLIP DIFFERENTIAL ===
        front_slip_avg = (wheel_slip_fl + wheel_slip_fr) / 2.0
        rear_slip_avg = (wheel_slip_rl + wheel_slip_rr) / 2.0
        slip_diff = rear_slip_avg - front_slip_avg
        
        # === APPLY OBJECTIVE LABELING ===
        label_str, label_int = calculate_label(slip_diff)
        
        # Update counters
        session_labels[label_str] += 1
        current_lap_labels[label_str] += 1
        
        # === DETECT LAP COMPLETION ===
        if laps > lapcount:
            lapcount = laps
            current_lap_labels = {'Neutral': 0, 'Understeer': 0, 'Oversteer': 0}
            ac.setText(l_status, "Lap {} Complete!".format(lapcount - 1))
        
        # === UPDATE UI ===
        ac.setText(l_lapcount, "Laps: {}".format(lapcount))
        
        if lapcount > 0:
            ac.setText(l_status, "Monitoring Lap {}".format(lapcount))
        
        # Current behavior with emoji indicators
        if label_str == "Oversteer":
            behavior_text = "Current: 🔴 OVERSTEER"
        elif label_str == "Understeer":
            behavior_text = "Current: 🔵 UNDERSTEER"
        else:
            behavior_text = "Current: 🟢 Neutral"
        ac.setText(l_current_label, behavior_text)
        
        # Slip differential value
        ac.setText(l_slip_diff, "Slip Diff: {:.2f}".format(slip_diff))
        
        # Current lap distribution
        lap_text = "This Lap: N:{} U:{} O:{}".format(
            current_lap_labels['Neutral'],
            current_lap_labels['Understeer'],
            current_lap_labels['Oversteer']
        )
        ac.setText(l_lap_stats, lap_text)
        
        # Session total distribution with percentages
        total = sum(session_labels.values())
        if total > 0:
            session_text = "Session: N:{}({:.0f}%) U:{}({:.0f}%) O:{}({:.0f}%)".format(
                session_labels['Neutral'], (session_labels['Neutral']/total)*100,
                session_labels['Understeer'], (session_labels['Understeer']/total)*100,
                session_labels['Oversteer'], (session_labels['Oversteer']/total)*100
            )
        else:
            session_text = "Session: N:0 U:0 O:0"
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
        # Out-lap
        ac.setText(l_status, "Out-Lap (Not Monitoring)")
        ac.setText(l_lapcount, "Laps: 0 (Out-Lap)")


def acShutdown():
    """Log final statistics on shutdown"""
    total = sum(session_labels.values())
    if total > 0:
        ac.log("=== ML TRAINING SESSION COMPLETE ===")
        ac.log("Total frames: {}".format(total))
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
    
    ac.console("ML Training Monitor shutdown - check logs for session stats")
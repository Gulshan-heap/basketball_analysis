"""
Basketball Analysis - Streamlit Application
============================================
A full-featured web UI wrapping the basketball_analysis pipeline.
"""

import os
import sys
import tempfile
import time
import pickle
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import pandas as pd

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Basketball Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #e94560; font-size: 2.8rem; margin: 0; }
    .main-header p  { color: #a8b2c1; font-size: 1.1rem; margin-top: 0.5rem; }

    .stat-card {
        background: #1e2a3a;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border-left: 4px solid #e94560;
        margin-bottom: 0.5rem;
    }
    .stat-card h3 { color: #a8b2c1; font-size: 0.85rem; margin: 0 0 0.3rem; text-transform: uppercase; }
    .stat-card h2 { color: #ffffff; font-size: 2rem; margin: 0; }

    .team1-card { border-left-color: #e94560 !important; }
    .team2-card { border-left-color: #4a90e2 !important; }

    .section-header {
        color: #e94560;
        font-size: 1.3rem;
        font-weight: 700;
        border-bottom: 2px solid #e94560;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 0.8rem;
    }
    .stProgress > div > div { background-color: #e94560; }

    div[data-testid="stSidebarContent"] { background-color: #0f1923; }
    div[data-testid="stSidebarContent"] .stMarkdown { color: #a8b2c1; }
</style>
""", unsafe_allow_html=True)

# ── header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏀 Basketball Analysis</h1>
    <p>AI-powered player tracking · team detection · tactical view · speed & distance metrics</p>
</div>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("images/basketball_court.png", use_container_width=True)
    st.markdown("---")

    st.markdown("### ⚙️ Model Paths")
    player_model = st.text_input("Player Detector (.pt)",
                                 value="models/player_detector.pt")
    ball_model   = st.text_input("Ball Detector (.pt)",
                                 value="models/ball_detector_model.pt")
    court_model  = st.text_input("Court Keypoint Detector (.pt)",
                                 value="models/court_keypoint_detector.pt")

    st.markdown("### 👕 Team Jersey Colors")
    team1_color = st.text_input("Team 1 jersey description", value="white shirt")
    team2_color = st.text_input("Team 2 jersey description", value="dark blue shirt")

    st.markdown("### 🔢 Processing Options")
    use_stubs    = st.checkbox("Use cached stubs (faster)", value=True)
    stub_dir     = st.text_input("Stub directory", value="stubs")
    output_fps   = st.slider("Output video FPS", 12, 60, 24)

    st.markdown("### 📊 Visualisation Layers")
    show_players    = st.checkbox("Player tracks",         value=True)
    show_ball       = st.checkbox("Ball track",            value=True)
    show_keypoints  = st.checkbox("Court key-points",      value=True)
    show_ball_ctrl  = st.checkbox("Team ball control",     value=True)
    show_frame_num  = st.checkbox("Frame numbers",         value=True)
    show_passes     = st.checkbox("Passes & interceptions",value=True)
    show_speed      = st.checkbox("Speed & distance",      value=True)
    show_tactical   = st.checkbox("Tactical view overlay", value=True)

    st.markdown("---")
    st.caption("Basketball Analysis App · github.com/Gulshan-heap/basketball_analysis")

# ── helper ─────────────────────────────────────────────────────────────────────
def check_models():
    missing = []
    for label, path in [
        ("Player detector",         player_model),
        ("Ball detector",           ball_model),
        ("Court keypoint detector", court_model),
    ]:
        if not Path(path).exists():
            missing.append(f"**{label}**: `{path}`")
    return missing


def load_stubs(stub_dir, video_stem):
    """Try to load all cached stubs for this video stem. Returns dict or None values."""
    stubs = {}
    for key in ["player_track", "ball_track", "court_key_points", "player_assignment"]:
        path = os.path.join(stub_dir, f"{key}_stubs.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                stubs[key] = pickle.load(f)
        else:
            stubs[key] = None
    return stubs


def frames_to_video(frames, output_path, fps=24):
    if not frames:
        return
    h, w = frames[0].shape[:2]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def compute_stats(ball_aquisition, player_assignment, passes, interceptions,
                  player_speed_per_frame, player_distances_per_frame):
    """Aggregate per-frame data into summary statistics."""
    total_frames = len(ball_aquisition)

    # Ball control %
    team1_ctrl = sum(1 for ba in ball_aquisition if ba != -1
                     and any(player_assignment[i].get(ba) == 1
                             for i in [ball_aquisition.index(ba)
                                       if ba in ball_aquisition else 0]))
    team_ctrl = {1: 0, 2: 0}
    for i, player_id in enumerate(ball_aquisition):
        if player_id == -1:
            continue
        team = player_assignment[i].get(player_id)
        if team in (1, 2):
            team_ctrl[team] += 1

    ctrl_total = team_ctrl[1] + team_ctrl[2]
    pct1 = round(100 * team_ctrl[1] / ctrl_total, 1) if ctrl_total else 0
    pct2 = round(100 * team_ctrl[2] / ctrl_total, 1) if ctrl_total else 0

    # Pass / interception counts
    team1_passes = sum(1 for p in passes if p == 1)
    team2_passes = sum(1 for p in passes if p == 2)
    team1_int    = sum(1 for p in interceptions if p == 1)
    team2_int    = sum(1 for p in interceptions if p == 2)

    # Per-player aggregates
    player_totals = {}      # player_id → {team, total_dist, max_speed, frames}
    for frame_i, speed_frame in enumerate(player_speed_per_frame):
        for pid, spd in speed_frame.items():
            if pid not in player_totals:
                player_totals[pid] = {"team": None, "total_dist_m": 0.0,
                                      "max_speed_kmh": 0.0, "frames": 0}
            if spd > player_totals[pid]["max_speed_kmh"]:
                player_totals[pid]["max_speed_kmh"] = spd
            player_totals[pid]["frames"] += 1

    for frame_i, dist_frame in enumerate(player_distances_per_frame):
        for pid, dist in dist_frame.items():
            if pid not in player_totals:
                player_totals[pid] = {"team": None, "total_dist_m": 0.0,
                                      "max_speed_kmh": 0.0, "frames": 0}
            player_totals[pid]["total_dist_m"] += dist

    # Attach team labels (use last seen assignment)
    for frame_assign in player_assignment:
        for pid, team in frame_assign.items():
            if pid in player_totals:
                player_totals[pid]["team"] = team

    return {
        "team_ctrl_pct": (pct1, pct2),
        "team_ctrl_frames": (team_ctrl[1], team_ctrl[2]),
        "passes": (team1_passes, team2_passes),
        "interceptions": (team1_int, team2_int),
        "player_totals": player_totals,
        "total_frames": total_frames,
    }


# ── sample video selector ─────────────────────────────────────────────────────
tab_upload, tab_sample = st.tabs(["📁 Upload Video", "🎬 Use Sample Video"])

video_path = None
with tab_upload:
    uploaded = st.file_uploader("Upload a basketball video",
                                type=["mp4", "avi", "mov", "mkv"])
    if uploaded:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()
        video_path = tmp.name
        st.video(video_path)

with tab_sample:
    sample_dir = "input_videos"
    samples = [f for f in os.listdir(sample_dir)
               if f.endswith((".mp4", ".avi", ".mov"))] if os.path.exists(sample_dir) else []
    if samples:
        chosen = st.selectbox("Choose a sample video", samples)
        if st.button("Use this sample", key="use_sample"):
            video_path = os.path.join(sample_dir, chosen)
            st.session_state["video_path"] = video_path
        if "video_path" in st.session_state and not uploaded:
            video_path = st.session_state["video_path"]
            st.video(video_path)
    else:
        st.info("No sample videos found in `input_videos/` directory.")

# ── main pipeline ──────────────────────────────────────────────────────────────
if video_path:
    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    with col_info:
        missing_models = check_models()
        if missing_models:
            st.warning("⚠️ Some model files are missing – analysis will fail without them:\n"
                       + "\n".join(missing_models))
        else:
            st.success("✅ All model files found. Ready to analyse.")

    if run_analysis:
        # ── import pipeline modules ─────────────────────────────────────────
        try:
            from utils import read_video, save_video
            from trackers import PlayerTracker, BallTracker
            from team_assigner import TeamAssigner
            from court_keypoint_detector import CourtKeypointDetector
            from ball_aquisition import BallAquisitionDetector
            from pass_and_interception_detector import PassAndInterceptionDetector
            from tactical_view_converter import TacticalViewConverter
            from speed_and_distance_calculator import SpeedAndDistanceCalculator
            from drawers import (
                PlayerTracksDrawer, BallTracksDrawer, CourtKeypointDrawer,
                TeamBallControlDrawer, FrameNumberDrawer, PassInterceptionDrawer,
                TacticalViewDrawer, SpeedAndDistanceDrawer,
            )
        except ImportError as e:
            st.error(f"Import error: {e}\nMake sure all dependencies are installed.")
            st.stop()

        progress_bar  = st.progress(0)
        status_text   = st.empty()

        def update(pct, msg):
            progress_bar.progress(pct)
            status_text.markdown(f"**{msg}**")

        with st.spinner("Running basketball analysis pipeline…"):
            t0 = time.time()

            # 1 – read frames
            update(5, "📽️ Reading video frames…")
            video_frames = read_video(video_path)
            total_frames = len(video_frames)

            # 2 – trackers
            update(10, "🔍 Initialising detectors…")
            player_tracker = PlayerTracker(player_model)
            ball_tracker   = BallTracker(ball_model)
            court_keypoint_detector = CourtKeypointDetector(court_model)

            os.makedirs(stub_dir, exist_ok=True)

            update(15, "🏃 Detecting & tracking players…")
            player_tracks = player_tracker.get_object_tracks(
                video_frames,
                read_from_stub=use_stubs,
                stub_path=os.path.join(stub_dir, "player_track_stubs.pkl"),
            )

            update(30, "🏀 Tracking ball…")
            ball_tracks = ball_tracker.get_object_tracks(
                video_frames,
                read_from_stub=use_stubs,
                stub_path=os.path.join(stub_dir, "ball_track_stubs.pkl"),
            )

            update(38, "🔑 Detecting court key-points…")
            court_keypoints_per_frame = court_keypoint_detector.get_court_keypoints(
                video_frames,
                read_from_stub=use_stubs,
                stub_path=os.path.join(stub_dir, "court_key_points_stub.pkl"),
            )

            # 3 – ball cleaning
            update(42, "🧹 Cleaning & interpolating ball track…")
            ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
            ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

            # 4 – team assignment
            update(48, "👕 Assigning players to teams…")
            team_assigner = TeamAssigner(
                team_1_class_name=team1_color,
                team_2_class_name=team2_color,
            )
            player_assignment = team_assigner.get_player_teams_across_frames(
                video_frames,
                player_tracks,
                read_from_stub=use_stubs,
                stub_path=os.path.join(stub_dir, "player_assignment_stub.pkl"),
            )

            # 5 – ball acquisition
            update(55, "🤝 Detecting ball possession…")
            ball_acq_detector = BallAquisitionDetector()
            ball_aquisition   = ball_acq_detector.detect_ball_possession(
                player_tracks, ball_tracks
            )

            # 6 – passes & interceptions
            update(60, "📊 Detecting passes & interceptions…")
            pi_detector   = PassAndInterceptionDetector()
            passes        = pi_detector.detect_passes(ball_aquisition, player_assignment)
            interceptions = pi_detector.detect_interceptions(ball_aquisition, player_assignment)

            # 7 – tactical view
            update(65, "🗺️ Computing tactical view…")
            tactical_converter = TacticalViewConverter(
                court_image_path="./images/basketball_court.png"
            )
            court_keypoints_per_frame = tactical_converter.validate_keypoints(
                court_keypoints_per_frame
            )
            tactical_player_positions = tactical_converter.transform_players_to_tactical_view(
                court_keypoints_per_frame, player_tracks
            )

            # 8 – speed & distance
            update(70, "⚡ Calculating speed & distance…")
            speed_calc = SpeedAndDistanceCalculator(
                tactical_converter.width,
                tactical_converter.height,
                tactical_converter.actual_width_in_meters,
                tactical_converter.actual_height_in_meters,
            )
            player_distances_per_frame = speed_calc.calculate_distance(tactical_player_positions)
            player_speed_per_frame     = speed_calc.calculate_speed(player_distances_per_frame)

            # 9 – draw output
            update(78, "🎨 Drawing visualisations…")
            output_video_frames = video_frames.copy()

            if show_players:
                drawer = PlayerTracksDrawer()
                output_video_frames = drawer.draw(
                    output_video_frames, player_tracks, player_assignment, ball_aquisition
                )
            if show_ball:
                drawer = BallTracksDrawer()
                output_video_frames = drawer.draw(output_video_frames, ball_tracks)
            if show_keypoints:
                drawer = CourtKeypointDrawer()
                output_video_frames = drawer.draw(output_video_frames, court_keypoints_per_frame)
            if show_frame_num:
                drawer = FrameNumberDrawer()
                output_video_frames = drawer.draw(output_video_frames)
            if show_ball_ctrl:
                drawer = TeamBallControlDrawer()
                output_video_frames = drawer.draw(
                    output_video_frames, player_assignment, ball_aquisition
                )
            if show_passes:
                drawer = PassInterceptionDrawer()
                output_video_frames = drawer.draw(
                    output_video_frames, passes, interceptions
                )
            if show_speed:
                drawer = SpeedAndDistanceDrawer()
                output_video_frames = drawer.draw(
                    output_video_frames, player_tracks,
                    player_distances_per_frame, player_speed_per_frame
                )
            if show_tactical:
                drawer = TacticalViewDrawer()
                output_video_frames = drawer.draw(
                    output_video_frames,
                    tactical_converter.court_image_path,
                    tactical_converter.width,
                    tactical_converter.height,
                    tactical_converter.key_points,
                    tactical_player_positions,
                    player_assignment,
                    ball_aquisition,
                )

            # 10 – save video
            update(90, "💾 Saving output video…")
            out_video_path = "output_videos/output_video.mp4"
            frames_to_video(output_video_frames, out_video_path, fps=output_fps)

            update(100, "✅ Analysis complete!")
            elapsed = time.time() - t0
            st.success(f"Done in {elapsed:.1f}s  ({total_frames} frames processed)")

        # ── store results in session ──────────────────────────────────────
        st.session_state["results"] = {
            "out_video_path": out_video_path,
            "output_video_frames": output_video_frames,
            "ball_aquisition": ball_aquisition,
            "player_assignment": player_assignment,
            "passes": passes,
            "interceptions": interceptions,
            "player_speed_per_frame": player_speed_per_frame,
            "player_distances_per_frame": player_distances_per_frame,
            "player_tracks": player_tracks,
            "total_frames": total_frames,
        }

# ── results section ────────────────────────────────────────────────────────────
if "results" in st.session_state:
    r = st.session_state["results"]

    st.markdown('<div class="section-header">📹 Analysed Video</div>', unsafe_allow_html=True)
    if os.path.exists(r["out_video_path"]):
        st.video(r["out_video_path"])
        with open(r["out_video_path"], "rb") as vf:
            st.download_button(
                "⬇️ Download output video",
                data=vf,
                file_name="basketball_analysis_output.mp4",
                mime="video/mp4",
            )

    # ── statistics ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">📊 Match Statistics</div>', unsafe_allow_html=True)

    stats = compute_stats(
        r["ball_aquisition"], r["player_assignment"],
        r["passes"], r["interceptions"],
        r["player_speed_per_frame"], r["player_distances_per_frame"],
    )

    pct1, pct2 = stats["team_ctrl_pct"]
    p1, p2     = stats["passes"]
    i1, i2     = stats["interceptions"]

    # top row – ball control
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stat-card team1-card">
            <h3>🔴 Team 1 Ball Control</h3>
            <h2>{pct1}%</h2>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h3>Total Frames</h3>
            <h2>{r['total_frames']}</h2>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card team2-card">
            <h3>🔵 Team 2 Ball Control</h3>
            <h2>{pct2}%</h2>
        </div>""", unsafe_allow_html=True)

    # ball control bar
    if pct1 + pct2 > 0:
        bar_html = f"""
        <div style="margin:1rem 0; background:#1e2a3a; border-radius:8px; overflow:hidden; height:28px;">
            <div style="float:left;width:{pct1}%;background:#e94560;height:100%;
                        display:flex;align-items:center;justify-content:center;
                        color:white;font-size:0.8rem;font-weight:700;">{pct1}%</div>
            <div style="float:left;width:{pct2}%;background:#4a90e2;height:100%;
                        display:flex;align-items:center;justify-content:center;
                        color:white;font-size:0.8rem;font-weight:700;">{pct2}%</div>
        </div>"""
        st.markdown(bar_html, unsafe_allow_html=True)

    # passes & interceptions
    st.markdown('<div class="section-header">🤝 Passes & Interceptions</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    for col, label, value, cls in [
        (c1, "🔴 Team 1 Passes",        p1, "team1-card"),
        (c2, "🔴 Team 1 Interceptions", i1, "team1-card"),
        (c3, "🔵 Team 2 Passes",        p2, "team2-card"),
        (c4, "🔵 Team 2 Interceptions", i2, "team2-card"),
    ]:
        with col:
            st.markdown(f"""
            <div class="stat-card {cls}">
                <h3>{label}</h3>
                <h2>{value}</h2>
            </div>""", unsafe_allow_html=True)

    # ── per-player table ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">🏃 Player Performance</div>', unsafe_allow_html=True)
    pt = stats["player_totals"]
    if pt:
        rows = []
        for pid, data in sorted(pt.items()):
            team_label = f"🔴 Team 1" if data["team"] == 1 else (
                         f"🔵 Team 2" if data["team"] == 2 else "Unknown")
            rows.append({
                "Player ID":      pid,
                "Team":           team_label,
                "Distance (m)":   round(data["total_dist_m"], 2),
                "Max Speed (km/h)": round(data["max_speed_kmh"], 1),
                "Active Frames":  data["frames"],
            })
        df = pd.DataFrame(rows).sort_values("Distance (m)", ascending=False).reset_index(drop=True)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # chart
        st.markdown('<div class="section-header">📈 Distance by Player</div>',
                    unsafe_allow_html=True)
        chart_df = df[["Player ID", "Distance (m)", "Team"]].copy()
        chart_df["Player ID"] = chart_df["Player ID"].astype(str)

        # colour map
        colors = ["#e94560" if t == "🔴 Team 1" else "#4a90e2"
                  for t in chart_df["Team"]]
        import altair as alt
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X("Player ID:N", sort="-y", title="Player"),
                y=alt.Y("Distance (m):Q", title="Total Distance (m)"),
                color=alt.Color("Team:N",
                                scale=alt.Scale(
                                    domain=["🔴 Team 1", "🔵 Team 2"],
                                    range=["#e94560", "#4a90e2"]
                                )),
                tooltip=["Player ID", "Team", "Distance (m)", "Max Speed (km/h)"],
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    # ── ball control over time ─────────────────────────────────────────────
    st.markdown('<div class="section-header">⏱️ Ball Control Over Time</div>',
                unsafe_allow_html=True)
    ba = r["ball_aquisition"]
    pa = r["player_assignment"]
    timeline_rows = []
    window = 30
    for i in range(0, len(ba) - window, window):
        seg = ba[i:i+window]
        t1 = sum(1 for j, pid in enumerate(seg)
                 if pid != -1 and pa[i+j].get(pid) == 1)
        t2 = sum(1 for j, pid in enumerate(seg)
                 if pid != -1 and pa[i+j].get(pid) == 2)
        total = t1 + t2 or 1
        timeline_rows.append({
            "Frame": i,
            "Team 1 %": round(100 * t1 / total),
            "Team 2 %": round(100 * t2 / total),
        })

    if timeline_rows:
        tdf = pd.DataFrame(timeline_rows)
        tdf_melted = tdf.melt("Frame", var_name="Team", value_name="Ball Control %")
        chart2 = (
            alt.Chart(tdf_melted)
            .mark_area(opacity=0.7)
            .encode(
                x=alt.X("Frame:Q", title="Frame"),
                y=alt.Y("Ball Control %:Q", stack="normalize",
                        axis=alt.Axis(format="%", title="Ball Control")),
                color=alt.Color("Team:N",
                                scale=alt.Scale(
                                    domain=["Team 1 %", "Team 2 %"],
                                    range=["#e94560", "#4a90e2"]
                                )),
                tooltip=["Frame", "Team", "Ball Control %"],
            )
            .properties(height=220)
        )
        st.altair_chart(chart2, use_container_width=True)

    # ── frame explorer ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">🔎 Frame Explorer</div>', unsafe_allow_html=True)
    frames = r["output_video_frames"]
    if frames:
        max_f = len(frames) - 1
        frame_idx = st.slider("Select frame", 0, max_f, 0)
        frame_rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, caption=f"Frame {frame_idx} / {max_f}",
                 use_container_width=True)

        # per-frame info
        pid_holding = ba[frame_idx] if frame_idx < len(ba) else -1
        team_holding = pa[frame_idx].get(pid_holding, "?") if pid_holding != -1 else "–"
        pass_ev = r["passes"][frame_idx] if frame_idx < len(r["passes"]) else -1
        int_ev  = r["interceptions"][frame_idx] if frame_idx < len(r["interceptions"]) else -1

        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Ball held by", f"Player {pid_holding}" if pid_holding != -1 else "–")
        fc2.metric("Team in possession",
                   f"🔴 Team {team_holding}" if team_holding in (1, 2) else "–")
        fc3.metric("Event",
                   f"Pass (Team {pass_ev})" if pass_ev != -1
                   else (f"Interception (Team {int_ev})" if int_ev != -1 else "–"))

    # ── export stats CSV ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">⬇️ Export Data</div>', unsafe_allow_html=True)
    if pt:
        csv = df.to_csv(index=False)
        st.download_button(
            "Download player stats CSV",
            data=csv,
            file_name="player_stats.csv",
            mime="text/csv",
        )

else:
    # ── landing / feature overview when no video loaded ───────────────────
    st.markdown('<div class="section-header">🏆 Features</div>', unsafe_allow_html=True)
    feat_cols = st.columns(4)
    features = [
        ("🏃", "Player Tracking",    "ByteTrack-powered multi-player tracking across every frame."),
        ("🏀", "Ball Tracking",      "YOLO-based ball detection with interpolation & filtering."),
        ("👕", "Team Detection",     "Fashion-CLIP jersey colour classification for automatic team assignment."),
        ("🗺️", "Tactical View",     "Homography-based top-down court view with player positions."),
        ("⚡", "Speed & Distance",   "Per-player real-world speed (km/h) and total distance (m)."),
        ("🤝", "Pass Detection",     "Automatic pass & interception events with team attribution."),
        ("🔑", "Court Key-points",   "Neural-network court landmark detection for calibration."),
        ("📊", "Match Statistics",   "Aggregated ball control %, passes, interceptions, and player charts."),
    ]
    for i, (icon, title, desc) in enumerate(features):
        with feat_cols[i % 4]:
            st.markdown(f"""
            <div style="background:#1e2a3a;border-radius:10px;padding:1rem;
                        margin-bottom:0.8rem;min-height:130px;">
                <div style="font-size:2rem;">{icon}</div>
                <div style="color:#e94560;font-weight:700;margin:0.3rem 0;">{title}</div>
                <div style="color:#a8b2c1;font-size:0.85rem;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 🚀 Getting Started
    1. **Upload** a basketball video (or choose a sample from the *Use Sample Video* tab)
    2. Verify **model paths** in the sidebar match your downloaded `.pt` files
    3. Optionally adjust **team jersey** descriptions to match your footage
    4. Click **Run Analysis** and wait for the pipeline to complete
    5. Explore the **annotated video**, **statistics**, and **frame explorer** below
    """)
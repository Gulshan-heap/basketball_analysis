"""
Basketball Analysis — Streamlit Application
============================================
Full-featured web UI wrapping the basketball_analysis pipeline.
"""

import os, sys, tempfile, time, pickle, shutil, subprocess
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Basketball Analysis",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── fonts & root ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── page background ── */
.stApp { background: #0b0f19; }

/* ── sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0f1923 100%);
    border-right: 1px solid #1e2d3d;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stSlider,
section[data-testid="stSidebar"] .stSelectbox select {
    background: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
}
section[data-testid="stSidebar"] hr { border-color: #21262d; }

/* ── hero banner ── */
.hero {
    background: linear-gradient(135deg, #0d1117 0%, #0f2027 40%, #1a1a2e 100%);
    border: 1px solid #21262d;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    display: flex;
    align-items: center;
    gap: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(233,69,96,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-icon { font-size: 4rem; line-height: 1; }
.hero-text h1 {
    color: #ffffff;
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0 0 0.3rem;
    letter-spacing: -0.5px;
}
.hero-text h1 span { color: #e94560; }
.hero-text p { color: #8b949e; font-size: 1rem; margin: 0; }
.hero-badges { display: flex; gap: 0.5rem; margin-top: 0.8rem; flex-wrap: wrap; }
.badge {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 20px;
    padding: 0.2rem 0.7rem;
    font-size: 0.72rem;
    color: #8b949e;
    font-weight: 600;
}

/* ── upload zone ── */
.upload-zone {
    background: #0d1117;
    border: 2px dashed #30363d;
    border-radius: 12px;
    padding: 2.5rem;
    text-align: center;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: #e94560; }

/* ── run button ── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #e94560, #c0392b) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.3px !important;
    box-shadow: 0 4px 15px rgba(233,69,96,0.3) !important;
    transition: all 0.2s !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(233,69,96,0.45) !important;
}

/* ── section heading ── */
.sec-head {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin: 2rem 0 1rem;
}
.sec-head-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #21262d, transparent);
}
.sec-head span {
    color: #e6edf3;
    font-size: 1.1rem;
    font-weight: 700;
    white-space: nowrap;
}
.sec-head-icon { font-size: 1.2rem; }

/* ── KPI cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom: 1rem; }
.kpi {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi::after {
    content: "";
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 0 0 12px 12px;
}
.kpi.red::after  { background: #e94560; }
.kpi.blue::after { background: #4a90e2; }
.kpi.gray::after { background: #30363d; }
.kpi.green::after { background: #3fb950; }
.kpi-label { color: #8b949e; font-size: 0.72rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 0.4rem; }
.kpi-value { color: #e6edf3; font-size: 2rem; font-weight: 800; line-height: 1; }
.kpi-sub   { color: #8b949e; font-size: 0.75rem; margin-top: 0.25rem; }

/* ── team comparison strip ── */
.team-strip {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 1rem;
}
.team-strip-header {
    display: flex;
    justify-content: space-between;
    padding: 0.9rem 1.2rem 0.4rem;
}
.team-strip-header .t1 { color: #e94560; font-weight: 700; font-size: 0.9rem; }
.team-strip-header .t2 { color: #4a90e2; font-weight: 700; font-size: 0.9rem; }
.team-strip-header .label { color: #8b949e; font-size: 0.8rem; }
.control-bar { height: 10px; display: flex; margin: 0 1.2rem 1rem; border-radius: 5px; overflow: hidden; }
.cb-t1 { background: linear-gradient(90deg, #e94560, #c0392b); }
.cb-t2 { background: linear-gradient(90deg, #3a7bd5, #4a90e2); }

/* ── stat row ── */
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 1.2rem;
    border-top: 1px solid #161b22;
}
.stat-row:nth-child(even) { background: #080c12; }
.stat-val { font-size: 1.1rem; font-weight: 700; }
.stat-val.red  { color: #e94560; }
.stat-val.blue { color: #4a90e2; }
.stat-name { color: #8b949e; font-size: 0.8rem; font-weight: 600; text-align: center; }

/* ── player table ── */
.stDataFrame { border: 1px solid #21262d !important; border-radius: 10px !important; }

/* ── frame explorer ── */
.frame-meta {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    display: flex;
    gap: 2rem;
    margin-top: 0.8rem;
}
.fm-item { display: flex; flex-direction: column; }
.fm-label { color: #8b949e; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.8px; }
.fm-value { color: #e6edf3; font-size: 1rem; font-weight: 700; }

/* ── feature card ── */
.feat-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.3rem;
    height: 100%;
    transition: border-color 0.2s, transform 0.2s;
}
.feat-card:hover { border-color: #e94560; transform: translateY(-2px); }
.feat-icon { font-size: 1.8rem; margin-bottom: 0.5rem; }
.feat-title { color: #e6edf3; font-size: 0.95rem; font-weight: 700; margin-bottom: 0.3rem; }
.feat-desc  { color: #8b949e; font-size: 0.78rem; line-height: 1.5; }

/* ── progress bar ── */
.stProgress > div > div > div > div { background: linear-gradient(90deg,#e94560,#c0392b) !important; }

/* ── tabs ── */
.stTabs [role="tab"] { color: #8b949e !important; font-weight: 600; }
.stTabs [role="tab"][aria-selected="true"] { color: #e94560 !important; border-bottom-color: #e94560 !important; }

/* ── alert overrides ── */
.stAlert { border-radius: 10px !important; }

/* ── video ── */
video { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def sec(icon, label):
    st.markdown(f"""
    <div class="sec-head">
        <span class="sec-head-icon">{icon}</span>
        <span>{label}</span>
        <div class="sec-head-line"></div>
    </div>""", unsafe_allow_html=True)


def kpi(label, value, sub="", colour="gray"):
    return f"""
    <div class="kpi {colour}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {"<div class='kpi-sub'>" + sub + "</div>" if sub else ""}
    </div>"""


def check_models(player_model, ball_model, court_model):
    missing = []
    for label, path in [("Player detector", player_model),
                        ("Ball detector", ball_model),
                        ("Court keypoint detector", court_model)]:
        if not Path(path).exists():
            missing.append(f"**{label}**: `{path}`")
    return missing


def frames_to_video(frames, output_path, fps=24):
    """Write frames → temp AVI → re-encode to H.264 mp4 via ffmpeg."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    tmp_avi = output_path.replace(".mp4", "_tmp.avi")
    writer = cv2.VideoWriter(tmp_avi, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()

    if shutil.which("ffmpeg"):
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", tmp_avi,
                 "-vcodec", "libx264", "-preset", "fast", "-crf", "23",
                 "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                 output_path],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            os.remove(tmp_avi)
            return
        except subprocess.CalledProcessError:
            pass
    os.replace(tmp_avi, output_path)


def compute_stats(ball_aquisition, player_assignment, passes, interceptions,
                  player_speed_per_frame, player_distances_per_frame):
    team_ctrl = {1: 0, 2: 0}
    for i, pid in enumerate(ball_aquisition):
        if pid == -1:
            continue
        team = player_assignment[i].get(pid)
        if team in (1, 2):
            team_ctrl[team] += 1

    ctrl_total = team_ctrl[1] + team_ctrl[2]
    pct1 = round(100 * team_ctrl[1] / ctrl_total, 1) if ctrl_total else 0
    pct2 = round(100 * team_ctrl[2] / ctrl_total, 1) if ctrl_total else 0

    p1 = sum(1 for p in passes if p == 1)
    p2 = sum(1 for p in passes if p == 2)
    i1 = sum(1 for p in interceptions if p == 1)
    i2 = sum(1 for p in interceptions if p == 2)

    player_totals = {}
    for speed_frame in player_speed_per_frame:
        for pid, spd in speed_frame.items():
            if pid not in player_totals:
                player_totals[pid] = {"team": None, "total_dist_m": 0.0, "max_speed_kmh": 0.0, "frames": 0}
            if spd > player_totals[pid]["max_speed_kmh"]:
                player_totals[pid]["max_speed_kmh"] = spd
            player_totals[pid]["frames"] += 1

    for dist_frame in player_distances_per_frame:
        for pid, dist in dist_frame.items():
            if pid not in player_totals:
                player_totals[pid] = {"team": None, "total_dist_m": 0.0, "max_speed_kmh": 0.0, "frames": 0}
            player_totals[pid]["total_dist_m"] += dist

    for frame_assign in player_assignment:
        for pid, team in frame_assign.items():
            if pid in player_totals:
                player_totals[pid]["team"] = team

    return {
        "team_ctrl_pct": (pct1, pct2),
        "team_ctrl_frames": (team_ctrl[1], team_ctrl[2]),
        "passes": (p1, p2),
        "interceptions": (i1, i2),
        "player_totals": player_totals,
        "total_frames": len(ball_aquisition),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.5rem;">
        <div style="font-size:2.5rem;">🏀</div>
        <div style="color:#e6edf3;font-weight:800;font-size:1.1rem;margin-top:0.3rem;">Basketball AI</div>
        <div style="color:#8b949e;font-size:0.72rem;margin-top:0.1rem;">Analysis Pipeline</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#21262d;margin:0.8rem 0;'>", unsafe_allow_html=True)

    with st.expander("🤖 Model Paths", expanded=True):
        player_model = st.text_input("Player Detector", value="models/player_detector.pt",
                                     label_visibility="visible")
        ball_model   = st.text_input("Ball Detector",   value="models/ball_detector_model.pt")
        court_model  = st.text_input("Court Keypoints", value="models/court_keypoint_detector.pt")

    with st.expander("👕 Team Jersey Colors", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div style='color:#e94560;font-size:0.75rem;font-weight:700;margin-bottom:4px;'>TEAM 1</div>", unsafe_allow_html=True)
            team1_color = st.text_input("T1", value="white shirt", label_visibility="collapsed")
        with col_b:
            st.markdown("<div style='color:#4a90e2;font-size:0.75rem;font-weight:700;margin-bottom:4px;'>TEAM 2</div>", unsafe_allow_html=True)
            team2_color = st.text_input("T2", value="dark blue shirt", label_visibility="collapsed")

    with st.expander("⚙️ Processing", expanded=False):
        use_stubs  = st.checkbox("Use cached stubs (faster)", value=True)
        stub_dir   = st.text_input("Stub directory", value="stubs")
        output_fps = st.slider("Output FPS", 12, 60, 24)

    with st.expander("🎨 Visualisation Layers", expanded=False):
        show_players   = st.checkbox("Player tracks",          value=True)
        show_ball      = st.checkbox("Ball track",             value=True)
        show_keypoints = st.checkbox("Court key-points",       value=True)
        show_ball_ctrl = st.checkbox("Team ball control",      value=True)
        show_frame_num = st.checkbox("Frame numbers",          value=True)
        show_passes    = st.checkbox("Passes & interceptions", value=True)
        show_speed     = st.checkbox("Speed & distance",       value=True)
        show_tactical  = st.checkbox("Tactical view overlay",  value=True)

    st.markdown("<hr style='border-color:#21262d;'>", unsafe_allow_html=True)
    st.markdown("<div style='color:#484f58;font-size:0.68rem;text-align:center;'>github.com/Gulshan-heap/basketball_analysis</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🏀</div>
    <div class="hero-text">
        <h1>Basketball <span>Analysis</span></h1>
        <p>AI-powered player tracking, team detection, tactical view &amp; performance metrics</p>
        <div class="hero-badges">
            <span class="badge">⚡ YOLO Detection</span>
            <span class="badge">🎯 ByteTrack</span>
            <span class="badge">👕 Fashion-CLIP</span>
            <span class="badge">🗺️ Homography</span>
            <span class="badge">📊 Real-time Stats</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO INPUT
# ─────────────────────────────────────────────────────────────────────────────
tab_upload, tab_sample = st.tabs(["📁  Upload Video", "🎬  Sample Video"])

video_path = None
with tab_upload:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Drop a basketball video here",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )
    if not uploaded:
        st.markdown("""
        <div class="upload-zone">
            <div style="font-size:2.5rem;margin-bottom:0.6rem;">📹</div>
            <div style="color:#e6edf3;font-weight:600;font-size:1rem;">Drop your video here</div>
            <div style="color:#8b949e;font-size:0.8rem;margin-top:0.3rem;">MP4 · AVI · MOV · MKV</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(uploaded.read())
        tmp.close()
        video_path = tmp.name
        st.video(video_path)

with tab_sample:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    sample_dir = "input_videos"
    samples = ([f for f in os.listdir(sample_dir) if f.endswith((".mp4", ".avi", ".mov"))]
               if os.path.exists(sample_dir) else [])
    if samples:
        chosen = st.selectbox("Choose a sample video", samples, label_visibility="collapsed")
        if st.button("▶ Load this sample", use_container_width=True):
            st.session_state["video_path"] = os.path.join(sample_dir, chosen)
        if "video_path" in st.session_state and not uploaded:
            video_path = st.session_state["video_path"]
            st.video(video_path)
    else:
        st.info("No sample videos found in `input_videos/` — add `.mp4` files there.")


# ─────────────────────────────────────────────────────────────────────────────
# RUN ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
if video_path:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    missing_models = check_models(player_model, ball_model, court_model)

    left, right = st.columns([1, 3])
    with left:
        run_analysis = st.button("🚀  Run Analysis", type="primary", use_container_width=True)
    with right:
        if missing_models:
            st.warning("⚠️ Missing models:\n" + "\n".join(missing_models))
        else:
            st.success("✅ All models found — ready to analyse")

    if run_analysis:
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
            st.error(f"Import error: {e}")
            st.stop()

        prog  = st.progress(0)
        status = st.empty()

        def upd(pct, msg):
            prog.progress(pct)
            status.markdown(f"<div style='color:#8b949e;font-size:0.85rem;margin-top:0.3rem;'>{msg}</div>",
                            unsafe_allow_html=True)

        t0 = time.time()

        upd(5,  "📽️ Reading video frames…")
        video_frames = read_video(video_path)
        total_frames = len(video_frames)

        upd(10, "🔍 Initialising detectors…")
        player_tracker = PlayerTracker(player_model)
        ball_tracker   = BallTracker(ball_model)
        court_kp_det   = CourtKeypointDetector(court_model)
        os.makedirs(stub_dir, exist_ok=True)

        upd(15, "🏃 Tracking players…")
        player_tracks = player_tracker.get_object_tracks(
            video_frames, read_from_stub=use_stubs,
            stub_path=os.path.join(stub_dir, "player_track_stubs.pkl"))

        upd(30, "🏀 Tracking ball…")
        ball_tracks = ball_tracker.get_object_tracks(
            video_frames, read_from_stub=use_stubs,
            stub_path=os.path.join(stub_dir, "ball_track_stubs.pkl"))

        upd(38, "🔑 Detecting court key-points…")
        court_kp = court_kp_det.get_court_keypoints(
            video_frames, read_from_stub=use_stubs,
            stub_path=os.path.join(stub_dir, "court_key_points_stub.pkl"))

        upd(42, "🧹 Cleaning ball track…")
        ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)
        ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

        upd(48, "👕 Assigning teams…")
        team_assigner = TeamAssigner(team_1_class_name=team1_color, team_2_class_name=team2_color)
        player_assignment = team_assigner.get_player_teams_across_frames(
            video_frames, player_tracks, read_from_stub=use_stubs,
            stub_path=os.path.join(stub_dir, "player_assignment_stub.pkl"))

        upd(55, "🤝 Detecting ball possession…")
        ball_acq_det  = BallAquisitionDetector()
        ball_aquisition = ball_acq_det.detect_ball_possession(player_tracks, ball_tracks)

        upd(60, "📊 Detecting passes & interceptions…")
        pi_det        = PassAndInterceptionDetector()
        passes        = pi_det.detect_passes(ball_aquisition, player_assignment)
        interceptions = pi_det.detect_interceptions(ball_aquisition, player_assignment)

        upd(65, "🗺️ Computing tactical view…")
        tactical_conv = TacticalViewConverter(court_image_path="./images/basketball_court.png")
        court_kp      = tactical_conv.validate_keypoints(court_kp)
        tactical_pos  = tactical_conv.transform_players_to_tactical_view(court_kp, player_tracks)

        upd(70, "⚡ Calculating speed & distance…")
        speed_calc = SpeedAndDistanceCalculator(
            tactical_conv.width, tactical_conv.height,
            tactical_conv.actual_width_in_meters, tactical_conv.actual_height_in_meters)
        dist_per_frame  = speed_calc.calculate_distance(tactical_pos)
        speed_per_frame = speed_calc.calculate_speed(dist_per_frame)

        upd(78, "🎨 Drawing visualisations…")
        out_frames = video_frames.copy()
        if show_players:
            out_frames = PlayerTracksDrawer().draw(out_frames, player_tracks, player_assignment, ball_aquisition)
        if show_ball:
            out_frames = BallTracksDrawer().draw(out_frames, ball_tracks)
        if show_keypoints:
            out_frames = CourtKeypointDrawer().draw(out_frames, court_kp)
        if show_frame_num:
            out_frames = FrameNumberDrawer().draw(out_frames)
        if show_ball_ctrl:
            out_frames = TeamBallControlDrawer().draw(out_frames, player_assignment, ball_aquisition)
        if show_passes:
            out_frames = PassInterceptionDrawer().draw(out_frames, passes, interceptions)
        if show_speed:
            out_frames = SpeedAndDistanceDrawer().draw(out_frames, player_tracks, dist_per_frame, speed_per_frame)
        if show_tactical:
            out_frames = TacticalViewDrawer().draw(
                out_frames, tactical_conv.court_image_path,
                tactical_conv.width, tactical_conv.height,
                tactical_conv.key_points, tactical_pos, player_assignment, ball_aquisition)

        upd(90, "💾 Encoding video (H.264)…")
        out_path = "output_videos/output_video.mp4"
        frames_to_video(out_frames, out_path, fps=output_fps)

        prog.progress(100)
        status.empty()

        elapsed = time.time() - t0
        st.success(f"✅ Analysis complete in **{elapsed:.1f}s** · {total_frames} frames processed")

        st.session_state["results"] = {
            "out_video_path": out_path,
            "output_video_frames": out_frames,
            "ball_aquisition": ball_aquisition,
            "player_assignment": player_assignment,
            "passes": passes,
            "interceptions": interceptions,
            "player_speed_per_frame": speed_per_frame,
            "player_distances_per_frame": dist_per_frame,
            "player_tracks": player_tracks,
            "total_frames": total_frames,
        }


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if "results" in st.session_state:
    r = st.session_state["results"]
    import altair as alt

    # ── output video ──────────────────────────────────────────────────────
    sec("📹", "Analysed Video")
    if os.path.exists(r["out_video_path"]):
        st.video(r["out_video_path"])
        with open(r["out_video_path"], "rb") as vf:
            st.download_button(
                "⬇️  Download MP4",
                data=vf,
                file_name="basketball_analysis.mp4",
                mime="video/mp4",
                use_container_width=False,
            )

    # ── stats ─────────────────────────────────────────────────────────────
    stats = compute_stats(
        r["ball_aquisition"], r["player_assignment"],
        r["passes"], r["interceptions"],
        r["player_speed_per_frame"], r["player_distances_per_frame"],
    )
    pct1, pct2 = stats["team_ctrl_pct"]
    p1, p2     = stats["passes"]
    i1, i2     = stats["interceptions"]
    pt         = stats["player_totals"]

    sec("📊", "Match Overview")
    st.markdown(f"""
    <div class="kpi-grid">
        {kpi("Team 1 Ball Control", f"{pct1}%", f"{stats['team_ctrl_frames'][0]} frames", "red")}
        {kpi("Team 2 Ball Control", f"{pct2}%", f"{stats['team_ctrl_frames'][1]} frames", "blue")}
        {kpi("Total Frames", f"{r['total_frames']:,}", "", "gray")}
        {kpi("Total Players", str(len(pt)), "tracked", "green")}
    </div>
    """, unsafe_allow_html=True)

    # ball control bar
    if pct1 + pct2 > 0:
        st.markdown(f"""
        <div class="team-strip">
            <div class="team-strip-header">
                <div class="t1">🔴 Team 1 — {pct1}%</div>
                <div class="label">BALL CONTROL</div>
                <div class="t2">{pct2}% — Team 2 🔵</div>
            </div>
            <div class="control-bar">
                <div class="cb-t1" style="width:{pct1}%"></div>
                <div class="cb-t2" style="width:{pct2}%"></div>
            </div>
            <div class="stat-row">
                <span class="stat-val red">{p1}</span>
                <span class="stat-name">PASSES</span>
                <span class="stat-val blue">{p2}</span>
            </div>
            <div class="stat-row">
                <span class="stat-val red">{i1}</span>
                <span class="stat-name">INTERCEPTIONS</span>
                <span class="stat-val blue">{i2}</span>
            </div>
            <div class="stat-row">
                <span class="stat-val red">{p1 + i1}</span>
                <span class="stat-name">TOTAL ACTIONS</span>
                <span class="stat-val blue">{p2 + i2}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── ball control over time ─────────────────────────────────────────────
    sec("⏱️", "Ball Control Over Time")
    ba, pa = r["ball_aquisition"], r["player_assignment"]
    timeline_rows, window = [], 30
    for i in range(0, len(ba) - window, window):
        seg = ba[i:i+window]
        t1 = sum(1 for j, pid in enumerate(seg) if pid != -1 and pa[i+j].get(pid) == 1)
        t2 = sum(1 for j, pid in enumerate(seg) if pid != -1 and pa[i+j].get(pid) == 2)
        tot = t1 + t2 or 1
        timeline_rows.append({"frame": i, "Team 1": round(100*t1/tot), "Team 2": round(100*t2/tot)})

    if timeline_rows:
        tdf = pd.DataFrame(timeline_rows).melt("frame", var_name="team", value_name="pct")
        chart_timeline = (
            alt.Chart(tdf)
            .mark_area(opacity=0.75, interpolate="monotone")
            .encode(
                x=alt.X("frame:Q", title="Frame"),
                y=alt.Y("pct:Q", stack="normalize",
                        axis=alt.Axis(format="%", title="Ball Control")),
                color=alt.Color("team:N",
                    scale=alt.Scale(domain=["Team 1","Team 2"], range=["#e94560","#4a90e2"]),
                    legend=alt.Legend(orient="top-right")),
                tooltip=["frame:Q", "team:N", "pct:Q"],
            )
            .properties(height=200)
            .configure_view(strokeWidth=0)
            .configure_axis(
                gridColor="#21262d", labelColor="#8b949e",
                titleColor="#8b949e", domainColor="#21262d"
            )
            .configure_legend(labelColor="#8b949e", titleColor="#8b949e",
                              fillColor="#0d1117", strokeColor="#21262d", padding=8)
        )
        st.altair_chart(chart_timeline, use_container_width=True)

    # ── player performance ─────────────────────────────────────────────────
    if pt:
        sec("🏃", "Player Performance")
        rows = []
        for pid, data in sorted(pt.items()):
            team_label = "🔴 Team 1" if data["team"] == 1 else ("🔵 Team 2" if data["team"] == 2 else "Unknown")
            rows.append({
                "Player ID":        pid,
                "Team":             team_label,
                "Distance (m)":     round(data["total_dist_m"], 2),
                "Max Speed (km/h)": round(data["max_speed_kmh"], 1),
                "Active Frames":    data["frames"],
            })
        df = pd.DataFrame(rows).sort_values("Distance (m)", ascending=False).reset_index(drop=True)
        st.dataframe(
            df.style
              .background_gradient(subset=["Distance (m)"],  cmap="RdYlGn")
              .background_gradient(subset=["Max Speed (km/h)"], cmap="RdYlGn"),
            use_container_width=True,
            hide_index=True,
        )

        # distance chart
        sec("📈", "Distance by Player")
        cdf = df.rename(columns={"Player ID":"pid","Distance (m)":"dist","Team":"team","Max Speed (km/h)":"spd"})
        cdf["pid"] = cdf["pid"].astype(str)
        chart_dist = (
            alt.Chart(cdf)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("pid:N", sort="-y", title="Player ID",
                        axis=alt.Axis(labelAngle=0, labelColor="#8b949e", titleColor="#8b949e")),
                y=alt.Y("dist:Q", title="Distance (m)",
                        axis=alt.Axis(labelColor="#8b949e", titleColor="#8b949e", gridColor="#21262d")),
                color=alt.Color("team:N",
                    scale=alt.Scale(domain=["🔴 Team 1","🔵 Team 2"], range=["#e94560","#4a90e2"]),
                    legend=alt.Legend(orient="top-right")),
                tooltip=[
                    alt.Tooltip("pid:N",  title="Player"),
                    alt.Tooltip("team:N", title="Team"),
                    alt.Tooltip("dist:Q", title="Distance (m)", format=".2f"),
                    alt.Tooltip("spd:Q",  title="Max Speed (km/h)", format=".1f"),
                ],
            )
            .properties(height=280)
            .configure_view(strokeWidth=0, fill="#0d1117")
            .configure_axis(domainColor="#21262d")
            .configure_legend(labelColor="#8b949e", titleColor="#8b949e",
                              fillColor="#0d1117", strokeColor="#21262d", padding=8)
        )
        st.altair_chart(chart_dist, use_container_width=True)

        # speed chart
        sec("⚡", "Max Speed by Player")
        chart_spd = (
            alt.Chart(cdf)
            .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
            .encode(
                x=alt.X("pid:N", sort="-y", title="Player ID",
                        axis=alt.Axis(labelAngle=0, labelColor="#8b949e", titleColor="#8b949e")),
                y=alt.Y("spd:Q", title="Max Speed (km/h)",
                        axis=alt.Axis(labelColor="#8b949e", titleColor="#8b949e", gridColor="#21262d")),
                color=alt.Color("team:N",
                    scale=alt.Scale(domain=["🔴 Team 1","🔵 Team 2"], range=["#e94560","#4a90e2"]),
                    legend=None),
                tooltip=[
                    alt.Tooltip("pid:N",  title="Player"),
                    alt.Tooltip("spd:Q",  title="Max Speed (km/h)", format=".1f"),
                ],
            )
            .properties(height=230)
            .configure_view(strokeWidth=0, fill="#0d1117")
            .configure_axis(domainColor="#21262d")
        )
        st.altair_chart(chart_spd, use_container_width=True)

    # ── frame explorer ─────────────────────────────────────────────────────
    sec("🔎", "Frame Explorer")
    frames = r["output_video_frames"]
    if frames:
        max_f     = len(frames) - 1
        frame_idx = st.slider("Scrub through frames", 0, max_f, 0, label_visibility="collapsed")
        frame_rgb = cv2.cvtColor(frames[frame_idx], cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, use_container_width=True)

        pid_holding  = ba[frame_idx] if frame_idx < len(ba) else -1
        team_holding = pa[frame_idx].get(pid_holding, "?") if pid_holding != -1 else "–"
        pass_ev  = r["passes"][frame_idx]        if frame_idx < len(r["passes"]) else -1
        int_ev   = r["interceptions"][frame_idx] if frame_idx < len(r["interceptions"]) else -1

        event_str = ("🟢 Pass (T" + str(pass_ev) + ")" if pass_ev != -1
                     else ("🟠 Interception (T" + str(int_ev) + ")" if int_ev != -1 else "—"))
        team_str  = (f"🔴 Team 1" if team_holding == 1
                     else ("🔵 Team 2" if team_holding == 2 else "—"))

        st.markdown(f"""
        <div class="frame-meta">
            <div class="fm-item"><div class="fm-label">Frame</div><div class="fm-value">{frame_idx} / {max_f}</div></div>
            <div class="fm-item"><div class="fm-label">Ball held by</div><div class="fm-value">{"Player " + str(pid_holding) if pid_holding != -1 else "—"}</div></div>
            <div class="fm-item"><div class="fm-label">Team in possession</div><div class="fm-value">{team_str}</div></div>
            <div class="fm-item"><div class="fm-label">Event</div><div class="fm-value">{event_str}</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── export ────────────────────────────────────────────────────────────
    if pt:
        sec("⬇️", "Export")
        ecol1, ecol2 = st.columns(2)
        with ecol1:
            st.download_button(
                "📄  Download Player Stats CSV",
                data=df.to_csv(index=False),
                file_name="player_stats.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with ecol2:
            summary = {
                "team1_ball_control_pct": pct1,
                "team2_ball_control_pct": pct2,
                "team1_passes": p1, "team2_passes": p2,
                "team1_interceptions": i1, "team2_interceptions": i2,
                "total_frames": r["total_frames"],
            }
            st.download_button(
                "📋  Download Match Summary JSON",
                data=pd.Series(summary).to_json(),
                file_name="match_summary.json",
                mime="application/json",
                use_container_width=True,
            )

else:
    # ── landing page ──────────────────────────────────────────────────────
    sec("🏆", "What This App Does")
    features = [
        ("🏃", "Player Tracking",     "ByteTrack-powered multi-player detection and tracking across every frame."),
        ("🏀", "Ball Tracking",       "YOLO-based ball detection with interpolation for missing frames."),
        ("👕", "Team Detection",      "Fashion-CLIP classifies jersey colours to auto-assign teams."),
        ("🗺️", "Tactical View",      "Homography maps all players onto a top-down court diagram."),
        ("⚡", "Speed & Distance",    "Real-world speed (km/h) and total distance (m) per player."),
        ("🤝", "Pass Detection",      "Automatic pass & interception events attributed per team."),
        ("🔑", "Court Key-points",    "Neural-network landmark detection for court calibration."),
        ("📊", "Match Statistics",    "Ball control %, player charts, frame explorer & CSV export."),
    ]
    cols = st.columns(4)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 4]:
            st.markdown(f"""
            <div class="feat-card">
                <div class="feat-icon">{icon}</div>
                <div class="feat-title">{title}</div>
                <div class="feat-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    sec("🚀", "Getting Started")
    st.markdown("""
    <div style="background:#0d1117;border:1px solid #21262d;border-radius:12px;padding:1.5rem 2rem;color:#8b949e;line-height:2;">
        <span style="color:#e6edf3;font-weight:700;">1.</span> Upload a basketball video using the tab above<br>
        <span style="color:#e6edf3;font-weight:700;">2.</span> Verify model paths in the sidebar match your <code style="background:#161b22;padding:2px 6px;border-radius:4px;">.pt</code> files<br>
        <span style="color:#e6edf3;font-weight:700;">3.</span> Adjust team jersey descriptions if needed<br>
        <span style="color:#e6edf3;font-weight:700;">4.</span> Hit <strong style="color:#e94560;">Run Analysis</strong> and wait for the pipeline to finish<br>
        <span style="color:#e6edf3;font-weight:700;">5.</span> Explore the annotated video, stats charts, and frame explorer
    </div>
    """, unsafe_allow_html=True)
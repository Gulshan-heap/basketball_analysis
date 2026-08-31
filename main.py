from utils import read_video,save_video
from trackers import PlayerTracker
from drawers import(
    PlayerTracksDrawer
)

def main():

    # Read video
    video_frames = read_video("input_videos/video_1.mp4")

    # Initialize Tracker
    player_tracker = PlayerTracker("models/player_detector.pt")

    # Run trackers
    player_tracks = player_tracker.get_object_tracks(video_frames,
                                                     read_from_stub=True,
                                                     stub_path="stubs/player_track_stubs.pkl")

    print(player_tracks)

    # draw output
    player_tracks_drawer = PlayerTracksDrawer()

    # Draw Object Tracks
    output_video_frames = player_tracks_drawer.draw(video_frames,player_tracks)

    # Save video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()

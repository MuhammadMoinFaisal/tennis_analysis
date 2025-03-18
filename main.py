from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from court_line_detector import CourtLineDetector
def main():
    #Read the Input Video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)

    #Detect and Track Players
    player_tracker = PlayerTracker(model_path = "yolo12n.pt")
    player_detections = player_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/player_detection.pkl")

    #Detect Tennis Ball
    ball_tracker = BallTracker(model_path = "models/tennis_ball_best.pt")
    ball_detections = ball_tracker.detect_frames(video_frames, read_from_stub=True, stub_path="tracker_stubs/ball_detection.pkl")

    #Detect Tennis Court Keypoints
    court_model_path = "models/keypoints_model_50.pth"
    court_line_detector = CourtLineDetector(court_model_path)
    court_keypoints = court_line_detector.predict(video_frames[0])

    #Choose & Filter Players
    player_detections = player_tracker.choose_and_filter_players(court_keypoints, player_detections)

    #Draw Output
    #Draw Player Bounding Boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)
    #Draw Tennis Ball Bounding Boxes
    output_video_frames = ball_tracker.draw_bboxes(output_video_frames, ball_detections)
    #Draw Tennis Court Keypoints
    output_video_frames = court_line_detector.draw_keypoints_on_video(output_video_frames, court_keypoints)

    #Save Video
    save_video(output_video_frames, "output_videos/output.avi")

if __name__ == "__main__":
    main()
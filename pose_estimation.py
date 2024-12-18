import cv2
import numpy as np

# Define body parts and pose pairs
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Model input dimensions
inWidth = 368
inHeight = 368

# Load the pre-trained model
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Confidence threshold
thres = 0.2

# Pose detection function
def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # Preprocess the image
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # Extract keypoints (19 body parts)

    assert len(BODY_PARTS) == out.shape[1]

    points = []
    for i in range(len(BODY_PARTS)):
        # Get heatmap for each body part
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = int((frameWidth * point[0]) / out.shape[3])
        y = int((frameHeight * point[1]) / out.shape[2])

        # Append keypoint only if confidence > threshold
        points.append((x, y) if conf > thres else None)

    # Draw the skeleton
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert partFrom in BODY_PARTS
        assert partTo in BODY_PARTS

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.circle(frame, points[idFrom], 3, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, points[idTo], 3, (0, 0, 255), cv2.FILLED)

    return frame


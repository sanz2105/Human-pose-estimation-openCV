import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

# Set default demo image
DEMO_IMAGE = 'stand.jpg'

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

# Set model parameters
width, height = 368, 368
inWidth, inHeight = width, height

# Streamlit interface
st.title("Human Pose Estimation with OpenCV")
st.text('Make sure you upload a clear image with all body parts visible.')

# File uploader
img_file_buffer = st.file_uploader(
    "Upload an image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"]
)

# Handle demo image if no file is uploaded
if img_file_buffer is not None:
    # Read uploaded image
    image = Image.open(img_file_buffer)
else:
    if os.path.exists(DEMO_IMAGE):
        image = Image.open(DEMO_IMAGE)
    else:
        st.error(f"Demo image '{DEMO_IMAGE}' not found. Please upload an image.")
        st.stop()

# Display original image
st.subheader("Original Image")
st.image(image, caption="Original Image", use_column_width=True)

# Threshold slider for detection
thres = st.slider(
    'Threshold for detecting key points (0-100%)', min_value=0, value=20, max_value=100, step=5
)
thres = thres / 100.0  # Convert to percentage (0-1 scale)

# Pose detection function
@st.cache(hash_funcs={np.ndarray: lambda x: str(x.shape)}, allow_output_mutation=True)
def poseDetector(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    # Load the neural network
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    assert len(BODY_PARTS) == out.shape[1]

    points = []
    for i in range(len(BODY_PARTS)):
        # Slice heatmap for the body part
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        # Add point only if confidence > threshold
        points.append((int(x), int(y)) if conf > thres else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert partFrom in BODY_PARTS
        assert partTo in BODY_PARTS

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    return frame

# Perform pose detection on the image
output = poseDetector(np.array(image))

# Display the output
st.subheader('Pose Estimated')
st.image(output, caption="Pose Estimation", use_column_width=True)

# Option to download the output image
output_path = "output_image.png"
cv2.imwrite(output_path, output)
st.download_button('Download Pose Estimated Image', output_path, file_name='pose_estimation.png')

st.markdown('---')

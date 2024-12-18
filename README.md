# Human Pose Estimation with OpenCV

This project implements a **Human Pose Estimation** system using OpenCV and TensorFlow. The system detects human body parts and visualizes their skeletons on an image. The app is built using **Streamlit** for the frontend, providing an interactive interface to upload images and visualize the pose estimation output.

## Features
- **Pose Estimation**: Detects and visualizes human body parts and their relationships (skeleton) using a pre-trained pose estimation model.
- **Threshold Control**: Allows users to set a threshold for detecting body keypoints.
- **Image Upload**: Users can upload images for pose estimation.
- **Image Download**: After processing, users can download the pose-estimated image.
- **Demo Image**: If no image is uploaded, a default demo image will be used for testing.

## Requirements
- Python 3.x
- OpenCV
- TensorFlow
- Streamlit
- NumPy
- PIL (Pillow)

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/sanz2105/Human-pose-estimation-openCV.git
   cd Human-pose-estimation-openCV
   ```

2. **Install Dependencies**:

   It is recommended to use a virtual environment to avoid conflicts with other projects.

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   pip install -r requirements.txt
   ```

3. **Download the Pre-trained Model**:

   The project uses the pre-trained pose estimation model `graph_opt.pb`. You can download it from [this link](https://github.com/sanz2105/Human-pose-estimation-openCV/releases) or use any compatible pre-trained model. Place it in the project directory.

### Running the App

1. **Start the Streamlit App**:

   To run the app locally, use the following command:

   ```bash
   streamlit run estimation_app.py
   ```

2. **Access the App**:

   Open your browser and navigate to `http://localhost:8501` to interact with the app.

## How to Use the App

1. **Upload Image**: Upload an image of a person (jpg/jpeg/png) by clicking on the "Upload Image" button.
2. **Adjust Threshold**: Use the slider to adjust the threshold for keypoint detection. This helps control how sensitive the pose detection is.
3. **View Pose Estimation**: Once the image is uploaded and processed, the estimated pose will be displayed with keypoints and skeleton lines drawn on it.
4. **Download Image**: After viewing the pose, you can download the image with the pose estimation drawn on it by clicking the "Download Pose Estimated Image" button.

## Files and Structure

- `estimation_app.py`: Main Streamlit app that handles file upload, pose detection, and image display.
- `pose_estimation.py`: Contains the pose detection logic using OpenCV’s deep learning module (`dnn`).
- `graph_opt.pb`: Pre-trained model file for pose estimation (must be downloaded and placed in the project directory).
- `stand.jpg`: Demo image used for testing the app when no image is uploaded.

## Contributing

Feel free to fork this repository and contribute by submitting pull requests. All contributions are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/sanz2105/Human-pose-estimation-openCV/blob/main/LICENSE) file for details.


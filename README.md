# Smart Fitness Assistant

My Project Smart Fitness Assistant offers real-time
workout tracking and personalized feedback.

This project combines Computer Vision and ML to
facilitate the user such that the system detects
exercises, monitors form & provides corrective feedback
ensuring proper technique during workouts.

This uses MediaPipe Pose to track body landmarks
in real-time while a trained Random Forest Classifier
accurately identifies the exercises.

Using trigonometric functions it analyzes key joint angles
and provides feedback on form & technique.

This ensures that each repetition is performed
with a proper technique, promoting optimal
results and reducing the risk of injury.

## Core components

- OpenCV - For video capture (Pose) & image processing.
- MediaPipe - For pose estimation.
- NumPy - For numerical operations.
- Scikit-learn - For ML (Random Forest Classifier).

## Usage/Example:

1. Install Dependencies
   Ensure you have Python installed, then install the required libraries:
   pip install opencv-python mediapipe numpy scikit-learn
   
2. Run the Application
   To start the real-time exercise tracking and form correction system, simply execute:
   python main.py
   
3. How It Works
   The system uses MediaPipe Pose to track body landmarks in real time.
   A Decision Tree Classifier predicts the performed exercise based on joint angles.
   It detects and counts repetitions for exercises like bicep curls, squats, tricep pushdown, bench press, and shoulder press.
   Real-time form feedback is provided to ensure correct technique and reduce injury risk.
   
4. Key Features
   - Real-time Exercise Recognition 
   - Automated Repetition Counting 
   - Pose Estimation & Joint Angle Analysis 
   - Instant Form Correction Feedback 

5. Quit the Application
  Press 'Q' to exit the live video feed.

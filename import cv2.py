import cv2
import mediapipe as mp
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Initialize variables for exercise counting
exercises = ['bicep_curls', 'squats', 'tricep_pushdown', 'bench_press', 'shoulder_press']
counters = {exercise: 0 for exercise in exercises}
stages = {exercise: None for exercise in exercises}

def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def extract_features(landmarks):
    # Extract relevant angles as features
    shoulder_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
    )
    elbow_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
    )
    knee_angle = calculate_angle(
        [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y],
        [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
    )
    return [shoulder_angle, elbow_angle, knee_angle]

# Simulate dataset creation (in a real scenario, you'd collect actual data)
def create_simulated_dataset():
    X = []
    y = []
    for _ in range(1000):  # Create 1000 simulated samples
        exercise = np.random.choice(exercises)
        if exercise == 'bicep_curls':
            features = [np.random.uniform(80, 160), np.random.uniform(0, 160), np.random.uniform(160, 180)]
        elif exercise == 'squats':
            features = [np.random.uniform(160, 180), np.random.uniform(160, 180), np.random.uniform(60, 160)]
        elif exercise == 'tricep_pushdown':
            features = [np.random.uniform(0, 60), np.random.uniform(60, 180), np.random.uniform(160, 180)]
        elif exercise == 'bench_press':
            features = [np.random.uniform(0, 60), np.random.uniform(60, 160), np.random.uniform(160, 180)]
        elif exercise == 'shoulder_press':
            features = [np.random.uniform(0, 60), np.random.uniform(0, 180), np.random.uniform(160, 180)]
        X.append(features)
        y.append(exercise)
    return X, y

# Train the machine learning model
def train_model():
    X, y = create_simulated_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    return clf

def check_form(exercise, angles):
    shoulder_angle, elbow_angle, knee_angle = angles
    feedback = ""

    if exercise == 'squats':
        if knee_angle > 170:
            feedback = "Squat deeper to engage muscles properly."
        elif knee_angle < 60:
            feedback = "Warning: Squatting too deep may strain knees."
        if shoulder_angle < 150:
            feedback += " Keep chest up and back straight."
    
    elif exercise == 'bench_press':
        if elbow_angle < 70:
            feedback = "Warning: Elbows too close to body. Risk of shoulder strain."
        elif elbow_angle > 110:
            feedback = "Keep elbows at about 90 degrees to protect shoulders."
    
    elif exercise == 'bicep_curls':
        if elbow_angle < 30 or elbow_angle > 160:
            feedback = "Maintain controlled movement. Don't swing weights."
    
    elif exercise == 'tricep_pushdown':
        if elbow_angle < 30:
            feedback = "Don't lock elbows at bottom. Risk of hyperextension."
        elif shoulder_angle > 30:
            feedback = "Keep elbows close to body for proper form."
    
    elif exercise == 'shoulder_press':
        if elbow_angle < 70 or elbow_angle > 170:
            feedback = "Maintain controlled movement. Don't lock elbows at top."
        if shoulder_angle > 30:
            feedback = "Keep core engaged. Don't arch back."

    return feedback

# Main function
def main():
    # Train the model
    clf = train_model()

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
          
            # Make detection
            results = pose.process(image)
        
            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                
                # Extract features and predict exercise
                features = extract_features(landmarks)
                predicted_exercise = clf.predict([features])[0]
                
                # Check form and get feedback
                form_feedback = check_form(predicted_exercise, features)
                
                # Update counters and stages based on the predicted exercise
                if predicted_exercise == 'bicep_curls':
                    angle = features[1]  # elbow angle
                    if angle > 160:
                        stages['bicep_curls'] = "down"
                    elif angle < 30 and stages['bicep_curls'] == 'down':
                        stages['bicep_curls'] = "up"
                        counters['bicep_curls'] += 1
                
                elif predicted_exercise == 'squats':
                    angle = features[2]  # knee angle
                    if angle > 160:
                        stages['squats'] = "up"
                    elif angle < 90 and stages['squats'] == 'up':
                        stages['squats'] = "down"
                        counters['squats'] += 1
                
                # Add similar logic for other exercises
                
                # Visualize
                cv2.putText(image, f"Exercise: {predicted_exercise}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Counter: {counters[predicted_exercise]}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Stage: {stages[predicted_exercise]}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display form feedback
                if form_feedback:
                    cv2.putText(image, "Form Check:", (10, 120), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                    # Split feedback into multiple lines if it's too long
                    words = form_feedback.split()
                    lines = []
                    current_line = []
                    for word in words:
                        if len(' '.join(current_line + [word])) < 40:
                            current_line.append(word)
                        else:
                            lines.append(' '.join(current_line))
                            current_line = [word]
                    if current_line:
                        lines.append(' '.join(current_line))
                    
                    for i, line in enumerate(lines):
                        cv2.putText(image, line, (10, 150 + i*30), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
                
            except:
                pass
            
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                     )               
            
            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
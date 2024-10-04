# Importing necessary modules
from flask import Flask, render_template, Response, request  # Flask for web application, render_template for rendering HTML templates, Response for handling HTTP responses, request for handling HTTP requests
import cv2  # OpenCV functions
import numpy as np  # Numerical operations
import mediapipe as mp  # Pose estimation

# Importing modules from mediapipe specifically for drawing utilities and pose estimation
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Creating a Flask application instance and setting the static URL path
app = Flask(__name__, static_url_path='/static')

# Defining a dictionary of weights and their corresponding calorie burn rates
weights = {
    '2.5 kg': 0.05,
    '5 kg': 0.1,
    '7.5 kg': 0.15,
    '10 kg': 0.2
}

# Initializing global variables counter and selected_weight
counter = 0
selected_weight = None

# Defining a function to calculate an angle given three points
def calangle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(rad * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Defining a function to generate a video feed with repetitions and stages
def generate_feed():
    # Open video capture device (camera)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    global counter, selected_weight  # Use global variables

    stage = None
    cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:  # Initialize Pose detection
        while cap.isOpened():  # Loop until camera is open
            ret, frame = cap.read()  # Read a frame from camera

            frame = cv2.flip(frame, 1)  # Flip frame horizontally to disable mirroring

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
            image.flags.writeable = False

            results = pose.process(image)  # Process the frame for pose estimation

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
            try:
                # Extract landmarks for left shoulder, elbow, and wrist
                landmarks = results.pose_landmarks.landmark
                shoulder = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calangle(shoulder, elbow, wrist)  # Calculate angle

                # Display angle on the frame
                cv2.putText(image, str(angle), tuple(np.multiply(elbow, [640, 480]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                if angle > 160:
                    stage = 'down'  # Arm is down
                if angle < 40 and stage == 'down':
                    stage = 'up'  # Arm is up, count as repetition
                    counter += 1

            except:
                pass

            # Display 'REPS' and 'STAGE' information on the frame
            cv2.rectangle(image, (0, 0), (150, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(image, (image.shape[1] - 225, 0), (image.shape[1], 73), (245, 117, 16), -1)
            cv2.putText(image, 'STAGE', (image.shape[1] - 160, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        cv2.LINE_AA)
            cv2.putText(image, stage, (image.shape[1] - 165, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2,
                        cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)  # Draw pose landmarks on the frame

            ret, jpeg = cv2.imencode('.jpg', image)  # Encode frame to jpg format
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # Yield frame in multipart format

    cap.release()  # Release video capture device

# Route for rendering index HTML template
@app.route('/')
def index():
    return render_template('index2.html')

# Route for starting the video feed
@app.route('/start_feed')
def start_feed():
    global counter  # Use global counter variable
    counter = 0  # Reset counter when starting the feed
    return Response(generate_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')  # Start the video feed and return as multipart response

# Route for resetting the counter
@app.route('/reset_counter')
def reset_counter():
    global counter  # Use global counter variable
    counter = 0  # Reset counter
    return 'Counter reset successfully.'  # Return success message

# Route for selecting weight
@app.route('/select_weight', methods=['POST'])
def select_weight():
    global selected_weight  # Use global selected_weight variable
    selected_weight = request.form['weight']  # Retrieve selected weight from form data
    return 'Weight selected successfully.'  # Return success message

# Route for calculating calories burned
@app.route('/calculate_calories', methods=['POST'])
def calculate_calories():
    global counter, selected_weight  # Use global counter and selected_weight variables
    if selected_weight and counter:  # Check if weight and counter are selected
        calories_burned = counter * weights[selected_weight]  # Calculate calories burned
        return f'{calories_burned} calories'  # Return calories burned
    else:
        return 'Please select weight and perform curls first.'  # Return error message if weight or counter is not selected

# Route for calculating BMI
@app.route('/calculate_bmi', methods=['POST'])
def calculate_bmi():
    height = float(request.form['height']) / 100  # Retrieve height from form data and convert to meters
    weight = float(request.form['weight'])  # Retrieve weight from form data

    if height > 0 and weight > 0:  # Check if height and weight are valid
        bmi = weight / (height ** 2)  # Calculate BMI
        return f'{bmi:.2f}'  # Return BMI with 2 decimal places
    else:
        return 'Invalid height or weight'  # Return error message if height or weight is invalid

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)  # Run the application in debug mode

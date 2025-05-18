import cv2
import csv
import os
import numpy as np
import speech_recognition as sr
import mediapipe as mp
import time
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1.0)  # Volume level (0.0 to 1.0)

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Constants
PIXELS_TO_INCHES = 0.0264583  # Calibration factor

def speak(text):
    """Make the AI speak the text and print it"""
    print("\nAI: " + text)
    engine.say(text)
    engine.runAndWait()
    time.sleep(1)

def listen():
    """Listen to user voice input with error handling"""
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("\n[DEBUG] Adjusting for ambient noise...")
        r.adjust_for_ambient_noise(source, duration=2)
        print("[DEBUG] Listening... (5 seconds)")
        try:
            speak("Please speak now")
            audio = r.listen(source, timeout=5, phrase_time_limit=5)
            print("[DEBUG] Processing audio...")
            try:
                text = r.recognize_google(audio).lower()
                print(f"[DEBUG] Recognized text: {text}")
                return text
            except sr.UnknownValueError:
                speak("Sorry, I couldn't understand what you said")
                return None
            except sr.RequestError as e:
                speak("There was an error with the speech service")
                print(f"[DEBUG] Recognition error: {e}")
                return None
        except sr.WaitTimeoutError:
            speak("I didn't hear anything. Please try again")
            print("[DEBUG] Listening timeout")
            return None

def get_user_gender():
    """Interactive conversation to determine user's gender"""
    print("\n" + "="*40)
    print("AI Body Measurement System")
    print("="*40)
    
    # Start conversation
    speak("Hello! Welcome to MeasureMe AI.")
    time.sleep(1)
    
    # Try voice recognition up to 3 times
    for attempt in range(3):
        speak("Please say clearly 'I am male' or 'I am female'.")
        
        voice_input = listen()
        
        if voice_input:
            print(f"You said: {voice_input}")
            
            # More comprehensive gender detection
            male_keywords = ['male', 'mail', 'mayl', 'mell', 'i am male', "i'm male", 'man']
            female_keywords = ['female', 'femal', 'femail', 'i am female', "i'm female", 'woman']
            
            male_score = sum(1 for word in male_keywords if word in voice_input)
            female_score = sum(1 for word in female_keywords if word in voice_input)
            
            print(f"[DEBUG] Male score: {male_score}, Female score: {female_score}")
            
            if male_score > female_score:
                speak("Thank you! I recognize you as male.")
                return "Male"
            elif female_score > male_score:
                speak("Thank you! I recognize you as female.")
                return "Female"
            else:
                speak("I'm not sure I understood. Let's try again.")
        else:
            speak("I didn't hear you clearly. Let's try again.")
        
        time.sleep(1)
    
    # Manual input fallback
    speak("I'm having trouble with voice recognition. Let's do this manually.")
    print("\nPlease select your gender manually:")
    while True:
        speak("Enter 1 for Male or 2 for Female")
        choice = input("Enter 1 for Male or 2 for Female: ").strip()
        if choice == "1":
            speak("You selected Male")
            return "Male"
        elif choice == "2":
            speak("You selected Female")
            return "Female"
        speak("Invalid input. Please enter 1 or 2")

def get_14_measurements(frame, gender):
    """Calculate all 14 body measurements"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    measurements = {}
    
    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        adjust = 1.07 if gender == "Male" else 0.93
        
        # 1. Head Circumference
        head = np.sqrt((lm[mp_pose.PoseLandmark.LEFT_EAR].x - lm[mp_pose.PoseLandmark.NOSE].x)**2 + 
                      (lm[mp_pose.PoseLandmark.LEFT_EAR].y - lm[mp_pose.PoseLandmark.NOSE].y)**2) * 3.14
        measurements["1. Head"] = round(head / PIXELS_TO_INCHES * adjust, 1)

        # 2. Neck Circumference
        neck = np.sqrt((lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x - lm[mp_pose.PoseLandmark.NOSE].x)**2 + 
                      (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y - lm[mp_pose.PoseLandmark.NOSE].y)**2) * 1.5
        measurements["2. Neck"] = round(neck / PIXELS_TO_INCHES * adjust, 1)

        # 3. Neck to Wrist
        neck_wrist = np.sqrt((lm[mp_pose.PoseLandmark.NOSE].x - lm[mp_pose.PoseLandmark.LEFT_WRIST].x)**2 + 
                           (lm[mp_pose.PoseLandmark.NOSE].y - lm[mp_pose.PoseLandmark.LEFT_WRIST].y)**2)
        measurements["3. Neck-Wrist"] = round(neck_wrist / PIXELS_TO_INCHES, 1)

        # 4. Shoulder Width
        shoulders = np.sqrt((lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)**2 + 
                          (lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y - lm[mp_pose.PoseLandmark.RIGHT_SHOULDER].y)**2)
        measurements["4. Shoulders"] = round(shoulders / PIXELS_TO_INCHES * adjust, 1)

        # 5. Chest (shoulders with expansion factor)
        measurements["5. Chest"] = round(shoulders * 1.1 / PIXELS_TO_INCHES * adjust, 1)

        # 6. Waist
        waist = np.sqrt((lm[mp_pose.PoseLandmark.LEFT_HIP].x - lm[mp_pose.PoseLandmark.RIGHT_HIP].x)**2 + 
                       (lm[mp_pose.PoseLandmark.LEFT_HIP].y - lm[mp_pose.PoseLandmark.RIGHT_HIP].y)**2) * 0.9
        measurements["6. Waist"] = round(waist / PIXELS_TO_INCHES * adjust, 1)

        # 7. Hips
        hips = np.sqrt((lm[mp_pose.PoseLandmark.LEFT_HIP].x - lm[mp_pose.PoseLandmark.RIGHT_HIP].x)**2 + 
                      (lm[mp_pose.PoseLandmark.LEFT_HIP].y - lm[mp_pose.PoseLandmark.RIGHT_HIP].y)**2)
        measurements["7. Hips"] = round(hips / PIXELS_TO_INCHES * adjust, 1)

        # 8. Neck to Waist
        neck_waist = np.sqrt((lm[mp_pose.PoseLandmark.NOSE].x - lm[mp_pose.PoseLandmark.LEFT_HIP].x)**2 + 
                           (lm[mp_pose.PoseLandmark.NOSE].y - lm[mp_pose.PoseLandmark.LEFT_HIP].y)**2)
        measurements["8. Neck-Waist"] = round(neck_waist / PIXELS_TO_INCHES, 1)

        # 9. Waist to Floor
        waist_floor = np.sqrt((lm[mp_pose.PoseLandmark.LEFT_HIP].x - lm[mp_pose.PoseLandmark.LEFT_ANKLE].x)**2 + 
                            (lm[mp_pose.PoseLandmark.LEFT_HIP].y - lm[mp_pose.PoseLandmark.LEFT_ANKLE].y)**2)
        measurements["9. Waist-Floor"] = round(waist_floor / PIXELS_TO_INCHES, 1)

        # 10. Waist to Knee
        waist_knee = np.sqrt((lm[mp_pose.PoseLandmark.LEFT_HIP].x - lm[mp_pose.PoseLandmark.LEFT_KNEE].x)**2 + 
                           (lm[mp_pose.PoseLandmark.LEFT_HIP].y - lm[mp_pose.PoseLandmark.LEFT_KNEE].y)**2)
        measurements["10. Waist-Knee"] = round(waist_knee / PIXELS_TO_INCHES, 1)

        # 11. Knee Circumference
        knee = np.sqrt((lm[mp_pose.PoseLandmark.LEFT_KNEE].x - lm[mp_pose.PoseLandmark.LEFT_KNEE].x)**2 + 
                      (lm[mp_pose.PoseLandmark.LEFT_KNEE].y - lm[mp_pose.PoseLandmark.LEFT_KNEE].y)**2) * 3.14
        measurements["11. Knee"] = round(knee / PIXELS_TO_INCHES * adjust, 1)

        # 12. Calf
        calf = np.sqrt((lm[mp_pose.PoseLandmark.LEFT_KNEE].x - lm[mp_pose.PoseLandmark.LEFT_ANKLE].x)**2 + 
                      (lm[mp_pose.PoseLandmark.LEFT_KNEE].y - lm[mp_pose.PoseLandmark.LEFT_ANKLE].y)**2) * 0.6
        measurements["12. Calf"] = round(calf / PIXELS_TO_INCHES, 1)

        # 13. Ankle
        ankle = np.sqrt((lm[mp_pose.PoseLandmark.LEFT_ANKLE].x - lm[mp_pose.PoseLandmark.LEFT_ANKLE].x)**2 + 
                       (lm[mp_pose.PoseLandmark.LEFT_ANKLE].y - lm[mp_pose.PoseLandmark.LEFT_ANKLE].y)**2) * 3.14
        measurements["13. Ankle"] = round(ankle / PIXELS_TO_INCHES * adjust, 1)

        # 14. Height
        height = np.sqrt((lm[mp_pose.PoseLandmark.NOSE].x - lm[mp_pose.PoseLandmark.LEFT_HEEL].x)**2 + 
                       (lm[mp_pose.PoseLandmark.NOSE].y - lm[mp_pose.PoseLandmark.LEFT_HEEL].y)**2)
        measurements["14. Height"] = round(height / PIXELS_TO_INCHES, 1)

        # Draw landmarks
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return frame, measurements

def main():
    gender = get_user_gender()
    
    cap = cv2.VideoCapture(0)
    speak(f"Measuring {gender} body... Please stand straight with your arms slightly away from your body.")
    print(f"\nMeasuring {gender} body... Stand in T-pose!")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame, measurements = get_14_measurements(frame, gender)
        
        # Display measurements
        y_offset = 30
        for name, value in measurements.items():
            cv2.putText(frame, f"{name}: {value} in", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_offset += 20
        
        cv2.imshow(f"AI MeasureMe ({gender})", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Save results
    if measurements:
        os.makedirs("output", exist_ok=True)
        with open("output/measurements.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Measurement", "Inches"])
            writer.writerows(measurements.items())
        speak("Measurements completed and saved successfully!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
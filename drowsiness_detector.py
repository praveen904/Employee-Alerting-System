import cv2
import numpy as np
from scipy.spatial import distance
import time
import logging
import os
import winsound

class DrowsinessDetector:
    def __init__(self):
        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Initialize logging
        logging.basicConfig(
            filename='drowsiness_events.log',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # Constants for drowsiness detection
        self.EAR_THRESHOLD = 0.20  # Lowered threshold to detect eye closure more easily
        self.CONSECUTIVE_FRAMES = 3  # Reduced number of frames needed to trigger alert
        self.ALERT_COOLDOWN = 5  # Reduced cooldown between alerts
        
        # Initialize counters and timers
        self.counter = 0
        self.last_alert_time = 0
        
        # Load alert sound
        self.alert_sound = "alert.wav"
        if not os.path.exists(self.alert_sound):
            # Create a simple alert sound if it doesn't exist
            self._create_alert_sound()

    def _create_alert_sound(self):
        """Create a simple alert sound file."""
        import wave
        import struct
        
        # Create a simple beep sound
        sampleRate = 44100
        duration = 1.0
        frequency = 440.0
        
        wavefile = wave.open(self.alert_sound, 'w')
        wavefile.setnchannels(1)
        wavefile.setsampwidth(2)
        wavefile.setframerate(sampleRate)
        
        for i in range(int(duration * sampleRate)):
            value = int(32767.0 * np.sin(frequency * np.pi * float(i) / float(sampleRate)))
            data = struct.pack('<h', value)
            wavefile.writeframes(data)
        
        wavefile.close()

    def calculate_ear(self, eye_points):
        """Calculate the Eye Aspect Ratio (EAR)."""
        # Compute the euclidean distances between the vertical eye landmarks
        A = distance.euclidean(eye_points[1], eye_points[5])
        B = distance.euclidean(eye_points[2], eye_points[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = distance.euclidean(eye_points[0], eye_points[3])
        
        # Calculate the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear

    def get_eye_points(self, eye_roi):
        """Get eye landmarks from eye region."""
        # Convert to grayscale
        gray_eye = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get binary image
        _, thresh = cv2.threshold(gray_eye, 70, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            cnt = max(contours, key=cv2.contourArea)
            
            # Get the extreme points
            leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
            rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])
            topmost = tuple(cnt[cnt[:,:,1].argmin()][0])
            bottommost = tuple(cnt[cnt[:,:,1].argmax()][0])
            
            # Calculate additional points for EAR
            mid_top = ((leftmost[0] + rightmost[0])//2, topmost[1])
            mid_bottom = ((leftmost[0] + rightmost[0])//2, bottommost[1])
            
            return np.array([leftmost, mid_top, rightmost, mid_bottom, bottommost, topmost])
        return None

    def process_frame(self, frame):
        """Process a single frame for drowsiness detection."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Region of interest for eyes
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                # Draw rectangle around eyes
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Get eye region
                eye_roi = roi_color[ey:ey+eh, ex:ex+ew]
                
                # Get eye points
                eye_points = self.get_eye_points(eye_roi)
                
                if eye_points is not None:
                    # Calculate EAR
                    ear = self.calculate_ear(eye_points)
                    
                    # Draw eye contours
                    cv2.drawContours(eye_roi, [eye_points], -1, (0, 255, 0), 1)
                    
                    # Check for drowsiness
                    if ear < self.EAR_THRESHOLD:
                        self.counter += 1
                        if self.counter >= self.CONSECUTIVE_FRAMES:
                            current_time = time.time()
                            if current_time - self.last_alert_time >= self.ALERT_COOLDOWN:
                                self._trigger_alert()
                                self.last_alert_time = current_time
                    else:
                        self.counter = 0
                    
                    # Display EAR value
                    cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display drowsiness status
                    status = "DROWSY" if self.counter >= self.CONSECUTIVE_FRAMES else "AWAKE"
                    cv2.putText(frame, f"Status: {status}", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Add visual indicator for eye closure
                    if ear < self.EAR_THRESHOLD:
                        cv2.putText(frame, "EYES CLOSED", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

    def _trigger_alert(self):
        """Trigger an alert when drowsiness is detected."""
        try:
            # Play a beep sound (frequency=1000Hz, duration=1000ms)
            winsound.Beep(1000, 1000)
            logging.info("Drowsiness detected - Alert triggered")
        except Exception as e:
            logging.error(f"Error playing alert sound: {str(e)}") 
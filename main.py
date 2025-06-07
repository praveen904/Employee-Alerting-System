import cv2
import sys
import os
from drowsiness_detector import DrowsinessDetector

def download_shape_predictor():
    """Download the facial landmark predictor if it doesn't exist."""
    import urllib.request
    import bz2
    
    predictor_path = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(predictor_path):
        print("Downloading facial landmark predictor...")
        url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
        urllib.request.urlretrieve(url, "shape_predictor_68_face_landmarks.dat.bz2")
        
        # Decompress the file
        with bz2.open("shape_predictor_68_face_landmarks.dat.bz2", 'rb') as source, \
             open(predictor_path, 'wb') as dest:
            dest.write(source.read())
        
        # Remove the compressed file
        os.remove("shape_predictor_68_face_landmarks.dat.bz2")
        print("Download complete!")

def main():
    # Download the facial landmark predictor if needed
    download_shape_predictor()
    
    # Initialize the drowsiness detector
    detector = DrowsinessDetector()
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        sys.exit(1)
    
    print("Starting drowsiness detection...")
    print("Press 'q' to quit")
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame")
            break
        
        # Process the frame
        processed_frame = detector.process_frame(frame)
        
        # Display the frame
        cv2.imshow('Drowsiness Detection', processed_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 
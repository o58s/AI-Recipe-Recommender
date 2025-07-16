from ultralytics import YOLO
import cv2 as cv
import json

# Load the trained YOLOv8 model from local path
model = YOLO(r'best.pt')

# Load the recipes data from JSON file
with open('recipes.json', 'r') as file:
    recipes = json.load(file)

# Initialize webcam capture (device 0)
cam = cv.VideoCapture(0)

print("📷 Press 'd' to detect ingredients and get recipe suggestions.")
print("❌ Press 'q' to quit.")

while True:
    # Capture frame-by-frame from webcam
    ret, frame = cam.read()

    # If frame not grabbed properly, exit the loop
    if not ret:
        print("Failed to grab frame")
        break
    
    # Show the raw webcam feed with instructions
    cv.imshow("Press 'd' to detect", frame)

    # Wait for key press for 1 ms
    key = cv.waitKey(1) & 0xFF

    # Quit if 'q' is pressed
    if key == ord('q'):
        print('Goodbye !!')
        break
    
    # If 'd' is pressed, run detection on current frame
    if key == ord('d'):
        # Perform prediction with the model
        result = model.predict(frame)

        # Get image with bounding boxes and labels drawn
        annotated_frame = result[0].plot()

        # Show the detection result window
        cv.imshow('Detection', annotated_frame)
        
        # Extract detected ingredient class names (unique)
        detected_ing = list(set([model.names[int(cls)] for cls in result[0].boxes.cls]))

        print("\n\n\n-------------------------------------")
        print('✅ detected ingredients: ', detected_ing)
        print('-------------------------------------\n\n\n')

        found = False  # Flag to check if any recipe matches

        # Check each recipe to see if all ingredients are detected
        for recipe in recipes:
            match = True

            for ingredient in recipe['ingredients']:
                if ingredient not in detected_ing:
                    match = False
                    break
            
            # If all ingredients matched, print recipe info
            if match:
                found = True
                print("✅", recipe["name"])
                print("   📝", recipe["instructions"])
                print("---------------------------------------------")
        
        # If no recipes matched, inform the user
        if not found:
            print("😕 No matching recipes found.")

# Release webcam and close all OpenCV windows
cam.release()
cv.destroyAllWindows()

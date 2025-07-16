from ultralytics import YOLO
import cv2 as cv
import json

model = YOLO(r'C:\Users\User\Desktop\AI Recipe Recommender from Fridge Images\best.pt')

with open('recipes.json', 'r') as file:
    recipes = json.load(file)

cam = cv.VideoCapture(0)

print("📷 Press 'd' to detect ingredients and get recipe suggestions.")
print("❌ Press 'q' to quit.")

while True:
    ret , frame =  cam.read()

    if not ret:
        print("Failed to grab frame")
        break
    
    cv.imshow("Press 'd' to detect" , frame)
    

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        print('Goodbye !!')
        break
    
    if key == ord('d'):
        result = model.predict(frame)

        annotated_frame = result[0].plot()

        cv.imshow('Detection', annotated_frame)
        

        detected_ing = list(set([model.names[int(cls)] for cls in result[0].boxes.cls]))


        print("\n\n\n-------------------------------------")
        print('✅ detected ingredients: ' , detected_ing)
        print('-------------------------------------\n\n\n')



        found = False

        for recipe in recipes:
            match = True

            for ingredient in recipe['ingredients']:
                if ingredient not in detected_ing:
                    match = False
                    break
        if match:
            found = True
            print("✅" ,  recipe["name"])
            print("   📝", recipe["instructions"])
            print("---------------------------------------------")
        
        if not found:
            print("😕 No matching recipes found.")

cam.release()
cv.destroyAllWindows()
from ultralytics import YOLO
import cv2


model_path = "model/yolov8n.pt"
model = YOLO(model_path)
# model = YOLO("yolov8n.pt")
class_name = {15: "cat",16: "dog",17: "horse", 18: "sheep",19: "cow",  20: "elephant",  21: "bear",  22: "zebra",  23: "giraffe"}

def get_prediction(img):
    global model
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = model.predict(img_rgb)
    threshold = 0.5
    count_text = 0
    animal_name = {}
    for result in results:
        for res in result.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = res
            print(score)
            if score > threshold and class_id in class_name:
                count_text += 1
                name = class_name[class_id]
                if name in animal_name:
                    animal_name[name] += 1
                else:
                    animal_name[name] = 1
                cv2.rectangle(img_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 6)
                cv2.putText(img_rgb, f"{class_name[class_id]} {class_id}", (int(x1), int(y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 2, cv2.LINE_AA)
    
    height, width, _ = img_rgb.shape
    ts = cv2.getTextSize("total="+str(count_text), cv2.FONT_HERSHEY_SIMPLEX, 1.3, 1)[0][0]
    x = width -70 - ts
    y = 50
    cv2.rectangle(img_rgb, (int(x-10), int(y-35)), (int(x + ts +30), int(y+15)), (255,0, 0), 4)
    cv2.putText(img_rgb, "total= "+ str(count_text), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255,0,0), 2, cv2.LINE_AA)
    return img_rgb,animal_name

# get_prediction(cv2.imread("test_img/image02.jpg")) 

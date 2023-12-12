import cv2
from ultralytics import YOLO 


# Open laptop camera stream
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    model = YOLO('best.pt')

    results = model(frame)
    # print(result)

    boxes_list = None
    conf_scores = None

    try:
        for result in results:
            # print(result.boxes)
            boxes_list = (result.boxes.xyxy)
            conf_scores = (result.boxes.conf)
            labels = (result.boxes.cls)

        boxes_list = boxes_list.tolist()
        conf_scores = conf_scores.tolist()
        labels = labels.tolist()

        
        for box, conf, labels in zip(boxes_list, conf_scores,labels):
            if conf > 0.5 and (labels == 1.0 or labels == 2.0):
                box = [int(x) for x in box]
                xmin, ymin = box[0], box[1]
                xmax, ymax = box[2], box[3]
        

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0) , 3)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 3
                font_thickness = 3
                text = 'Front is Detected'
                # width = frame.shape[0]
                # height = frame.shape[1]
                text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                text_position = (15,  15)  # Adjust the Y-coordinate to place the text above the rectangle
                cv2.putText(frame, text, text_position, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    except: 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_thickness = 3
        text = 'Please rotate front Detected'
                # width = frame.shape[0]
                # height = frame.shape[1]
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
        text_position = (300,  300)

        cv2.putText(frame, text, text_position, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
        cv2.imshow('Laptop Camera', frame)
        

    # Display the frame
    cv2.imshow('Laptop Camera', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

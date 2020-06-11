import cv2
import numpy as np
from keras.models import model_from_json


json_file = open('hindi_digits_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("hindi_digits_model.h5")
print("Loaded model from disk")


canvas = np.ones((500, 500), dtype="uint8") * 255
canvas[180:320,180:320] = 0

start_pt = None
end_pt = None
is_draw = False

def draw_line(img,start_at,end_at):
    cv2.line(img,start_at,end_at,255,15)

def on_events_mouse(event,x,y,flags,params):
    global start_pt
    global end_pt
    global canvas
    global is_draw
    if event == cv2.EVENT_LBUTTONDOWN:
        if is_draw:
            start_pt = (x,y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_draw:
            end_pt = (x,y)
            draw_line(canvas,start_pt,end_pt)
            start_pt = end_pt
    elif event == cv2.EVENT_LBUTTONUP:
        is_draw = False

def predict(image):
    input = cv2.resize(image, (130, 130)).reshape((1, 130, 130, 1)) / 255
    pred = loaded_model.predict(input)
    return ("Final output:{}".format(np.argmax(pred)))


cv2.namedWindow("Test Canvas")
cv2.setMouseCallback("Test Canvas", on_events_mouse)

while(True):
    cv2.imshow("Test Canvas", canvas)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        is_draw = True
    elif key == ord('c'):
        canvas[180:320,180:320] = 0
    elif key == ord('p'):
        image = canvas[180:320,180:320]
        result = predict(image)
        print(result)

cv2.destroyAllWindows()

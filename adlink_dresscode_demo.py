TEST_FILE = './helmet.jpg'
TF_LITE_MODEL = './ssd_mobilenet_v3_in-uint8_out-uint8_tensor_ptq.tflite'

LABEL_MAP = './label_info.txt'
THRESHOLD = 0.55
LABEL_SIZE = 1.0
RUNTIME_ONLY = True

import cv2
import numpy as np

if RUNTIME_ONLY:
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=TF_LITE_MODEL)
else:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=TF_LITE_MODEL)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

_, INPUT_HEIGHT, INPUT_WIDTH, _ = interpreter.get_input_details()[0]['shape']

#print("width:", INPUT_WIDTH)
#print("height:", INPUT_HEIGHT)

floating_model = input_details[0]['dtype'] == np.float32

#print("input_details:", input_details)

with open(LABEL_MAP, 'r') as f:
    labels = [line.strip() for line in f.readlines()]
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

img = cv2.imread(TEST_FILE, cv2.IMREAD_COLOR)
IMG_HEIGHT, IMG_WIDTH = img.shape[:2]

print("width:", IMG_WIDTH)
print("height:", IMG_HEIGHT)

pad = abs(IMG_WIDTH - IMG_HEIGHT) // 2
x_pad = pad if IMG_HEIGHT > IMG_WIDTH else 0
y_pad = pad if IMG_WIDTH > IMG_HEIGHT else 0
img_padded = cv2.copyMakeBorder(img, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad,
                                borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
IMG_HEIGHT, IMG_WIDTH = img_padded.shape[:2]

print("width:", IMG_WIDTH)
print("height:", IMG_HEIGHT)

img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)
img_r = cv2.resize(img_padded, (INPUT_WIDTH, INPUT_HEIGHT))
input_data = np.expand_dims(img_resized, axis=0)

if floating_model:
    input_data = np.float32(input_data - 127.5) / 127.5 

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

output_data = interpreter.get_tensor(output_details[0]['index'])
results = np.squeeze(output_data)
#for i in range(5):
#    print(results[i])
ind = np.argsort(results, axis=0)[::-1]
#print("output_details:", output_details)

im_w = IMG_WIDTH
im_h = IMG_HEIGHT
for j in range(len(results[ind[0]])):
    if j == 0: #skipping the background detection
        continue
    color = ((j-1)*100, 0, 200)
    for i in range(len(ind)):
        loc = (results[ind[i]][j]) #/ 255.0
        score = loc[6] / 255.0
        if score < THRESHOLD:
            continue
        (xmin,ymin,xmax,ymax) = loc[0:4] / 255.0 #INPUT_WIDTH
        (left,right,top,bottom) = (xmin * IMG_WIDTH, xmax * IMG_WIDTH, ymin * IMG_HEIGHT, ymax * IMG_HEIGHT)
        #print("result:", results[ind[i]][j])
        #print('Score:{},Loc:{},{},{},{}'.format(score, left, right, top, bottom))

        min_y = round(ymin * IMG_HEIGHT)
        min_x = round(xmin * IMG_WIDTH)
        max_y = round(ymax * IMG_HEIGHT)
        max_x = round(xmax * IMG_WIDTH)
        cv2.rectangle(img_padded, (min_x, min_y + y_pad), (max_x, max_y + y_pad), color, 5)

print('W:{}, H:{}, xpad:{}, ypad:{}'.format(IMG_WIDTH, IMG_HEIGHT, x_pad, y_pad))
img_show = img_padded[y_pad: IMG_HEIGHT - y_pad, x_pad: IMG_WIDTH - x_pad]
cv2.namedWindow('Object detection', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Object detection',
                 1024 if IMG_WIDTH > IMG_HEIGHT else round(1024 * IMG_WIDTH / IMG_HEIGHT),
                 1024 if IMG_HEIGHT > IMG_WIDTH else round(1024 * IMG_HEIGHT / IMG_WIDTH))
cv2.imshow('Object detection', img_show)
cv2.imwrite('./result.jpg', img_show)
cv2.waitKey(0)
cv2.destroyAllWindows()

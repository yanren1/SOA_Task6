# SPDX-License-Identifier: MIT
from PIL import Image,ImageDraw
import onnxruntime as ort
import numpy as np
from box_utils import predict


def scale(box):
    width = box[2] - box[0]
    height = box[3] - box[1]
    maximum = max(width, height)
    dx = int((maximum - width)/2)
    dy = int((maximum - height)/2)

    bboxes = [box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy]
    return bboxes

# face detection method
def faceDetector(orig_image, threshold = 0.7):
    face_detector_onnx = "version-RFB-320.onnx"
    face_detector = ort.InferenceSession(face_detector_onnx)
    image = orig_image.resize((320, 240))

    image_mean = np.array([127, 127, 127])
    image = (image - image_mean) / 128
    image = np.transpose(image, [2, 0, 1])
    image = np.expand_dims(image, axis=0)
    image = image.astype(np.float32)

    input_name = face_detector.get_inputs()[0].name
    confidences, boxes = face_detector.run(None, {input_name: image})
    boxes, labels, probs = predict(orig_image.width, orig_image.height, confidences, boxes, threshold)
    return boxes, labels, probs

def onnx_inf(img):
    orig_image = Image.open(img)
    boxes, labels, probs = faceDetector(orig_image)
    draw = ImageDraw.Draw(orig_image)

    for i in range(boxes.shape[0]):
        box = scale(boxes[i, :])
        draw.rectangle([box[0], box[1], box[2], box[3]], outline="red")

    # orig_image.show()
    return boxes, orig_image.tobytes()

if __name__ == '__main__':
    boxes,img = onnx_inf('1.jpg')
    print(boxes)
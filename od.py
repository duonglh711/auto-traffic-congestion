import numpy as np
import cv2
class ObjectDetection():
  def __init__(self, weights, names, configs, dsize, confidence, threshold):
    self.names = open(names).read().strip().split('\n')
    self.confidence = confidence
    self.threshold = threshold
    self.dsize = (dsize, dsize) if not isinstance(dsize, (tuple, list)) else dsize 
    self.color = np.random.randint(0, 255, size=(len(self.names), 3),
	dtype="uint8")
    self.load_net(weights, configs)
  def load_net(self, weights, configs):
    print("[INFO] Loading Object Detection ...")
    net = cv2.dnn.readNetFromDarknet(configs, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    self.net = net
    self.ln = ln
    print("[INFO] Loading Object Detection Done!")
  def predict(self, blob, base, margin):
    self.net.setInput(blob)
    layerOutputs = np.asarray(self.net.forward(self.ln))
    return self.extract_box(layerOutputs, base, margin)
  
  def extract_box(self, layerOutputs, base, margin):
    boxes = []
    confidences = []
    class_ids = []
    for output in layerOutputs:
      position = output[:, 0:4]
      score = output[:, 5:]
      satisfy = np.amax(score, axis=-1)
      pos = np.argmax(score, axis=-1)
      confidence = satisfy[satisfy>self.confidence].astype(np.float32).tolist()
      class_id = pos[satisfy>self.confidence].astype(np.int32).tolist()
      confidences.extend(confidence)
      class_ids.extend(class_id)
      boxe = (position[satisfy>self.confidence]*base)+margin
      boxe[:, 0] -= boxe[:, 2]//2
      boxe[:, 1] -= boxe[:, 3]//2
      boxes.extend(boxe.astype(np.int32).tolist())
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence, self.threshold)
    if len(idxs) > 0:
      idxs = idxs.flatten().astype(int)
      boxes = np.take(boxes, idxs, axis=0)
      confidences = np.take(confidences, idxs)
      class_ids = np.take(class_ids, idxs)
    return boxes, confidences, class_ids, idxs
  
  def draw_boxes(self, img, boxes, confidences, class_ids, idxs):
    for id,box in enumerate(boxes):
      (x, y) = (box[0], box[1])
      (w, h) = (box[2], box[3])
      # color = [int(c) for c in self.color[class_ids[id]]]
      cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
      # text = "{}: {:.3f}".format(self.names[class_ids[id]],confidences[id])
      # cv2.putText(img, '', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)









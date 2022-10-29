import os.path
import time
import tensorflow as tf
from absl import app, flags
from absl.flags import FLAGS
from tensorflow.python.saved_model import tag_constants
from tools import generate_detections as gdet
import cv2
import cv2 as cv
import numpy as np
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
# from tools import generate_detections as gdet
import density
from od import ObjectDetection
import math
from enum import Enum
# from multiprocessing import Process, Queue
from threading import Thread
from queue import Queue
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

flags.DEFINE_string('weights', './weights/v2_yolov4.weights','path to yolov4 weights file')
flags.DEFINE_string('names', './yolov4/custom.name','path to yolov4 obj name file')
flags.DEFINE_string('configs', './yolov4/custom.cfg','path to yolov4 config file')
flags.DEFINE_integer('size', 832, 'resize images to')
# /content/drive/MyDrive/ThesisFinal/TO_duong/video/QT-iPhone.mp4
flags.DEFINE_string('input', './video/SA.mp4', 'path to input video')
flags.DEFINE_string('output_folder', './outputs', 'path to output')
flags.DEFINE_string('output_format', 'XVID', 'codec used in VideoWriter when saving video to file')
flags.DEFINE_float('threshold', 0.5, 'iou threshold')
flags.DEFINE_float('confidence', 0.50, 'score threshold')
flags.DEFINE_string('crowd_weights', './weights/bnet_qt_6.h5', 'path to bnet weights')
flags.DEFINE_integer('bg_init', 500, 'number of background init')
flags.DEFINE_integer('max_frame', -1, 'number frame to detect, put -1 to detect all frame')
flags.DEFINE_bool('is_show', True, 'show preview')
flags.DEFINE_integer('start_minutes', 0, 'detect from minutes')
def init_regression(regres):
    lst = []
    for key, value in regres.items():
        lst.append(str(key) + '\t' + str(value[0]) + "\n")

    with open("regression_PVT2.txt", 'w') as f:
        f.writelines(lst)
        f.close()


def init_apply(algs):
  kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
  kernel4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
  def apply(frame):
    mask = algs.apply(frame)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel4)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel1)
    return mask
  return apply


def init_touchpoint(detections, bkg):
    length = len(detections)
    detect_obj = list(range(0, length))
    while len(detect_obj) > 0:
        detection = detections[detect_obj[0]]
        beta = 0.15
        x, y, xm, ym = detection.to_tlbr().astype(int)
        b = False
        if detection.class_id == 0:
            i = 1
            tlength = len(detect_obj) - 1
            while i < tlength:
                x_, y_, x_m, y_m = detections[detect_obj[i]].to_tlbr().astype(int)
                if not (xm < x_ or x_m < x or y > y_m or y_ > ym):
                    u, d, th, tw, tw_ = 0, i, ym, x, xm
                    if y_ > ym:
                        u, d, th, tw, tw_ = d, u, y_m, x_, x_m
                    score = np.sum(bkg[th // 4 - 1, tw // 4: tw_ // 4] / 255) / (tw_ - tw)
                    if score > 0.25:
                        b = True
                        detect_obj.pop(d)
                        tlength -= 1
                        continue
                i += 1
            if b:
                detect_obj.pop(0)
                continue

        x, y, w, h = (detection.tlwh // 4).astype(int)
        pos_x = int(x + w / 2) * 4
        if detection.class_id == 0:
            h = h + h * beta
            x, y, w, h = int(round(x + w // 6)), int(round(y)), int(round(x + 5 * w // 6)), int(round(y + h))
        else:
            w += x
            h = int(y + 1.1 * h)
        obj = bkg[y:h, x: w].astype(int)
        layer = np.ones((w - x,), dtype=int)
        length = obj.shape[0]
        count = length - 1
        point = None
        while count > 0:
            if np.sum(np.bitwise_and(layer, obj[count])) > 0:
                point = (np.where(obj[count] > 0)[0][0], count)
                break
            count -= 1
        if point:
            detection.touchPoint = (pos_x, int(y + point[1]) * 4)

        detect_obj.pop(0)
    return detections


def crowd_estimation(input_shape=(None, None, 3)):
    model = density.bnet.BNetv3(input_shape)
    model = density.handling.load_model(model, FLAGS.crowd_weights)
    return model


def background_execution(vid, out_shape, top, left, bottom, right, init=400):
    bkg = cv2.createBackgroundSubtractorMOG2(detectShadows=True, varThreshold=16)
    # bkg = cv2.bgsegm.createBackgroundSubtractorMOG(noiseSigma=10, history=int(init), nmixtures=5, backgroundRatio=0.5)
    # bkg = cv2.bgsegm.createBackgroundSubtractorGSOC(nSamples=3, propagationRate=0.05, hitsThreshold=16, noiseRemovalThresholdFacFG=0.0025)

    count = 0
    while count < init:
        count += 1
        return_val, frame = vid.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame[top:bottom, left:right], out_shape)
        bkg.apply(frame)
        print ("INIT: %d/%d"%(count, init))
    return bkg, None


def video_reader(vid, out_width=None, out_height=None):
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
    out_width = int(out_width * width) or width
    out_height = int(out_height * height) or height

    out = cv2.VideoWriter(get_output_path(OUTPUT_TYPE.VIDEO), codec, fps, (out_width, out_height))
    return out, width, height, fps


def region_of_interested(raw_rois, input_size, bkg_shape, width, height):
    roi_points = []
    for point in raw_rois:
        roi_points.append(( int(point[0]*width), int(point[1]*height)))
    return  roi_points, np.asarray(raw_rois)

def distance_calculation(distance_mapping=0.005):
    return lambda x, y: distance_mapping * abs(y - x)

def time_estimation(fps, is_time=False):
    if is_time:
        return lambda x, y: y - x
    else:
        return lambda x, y: (y - x) / fps

def speed_estimation( fps=30):
    return lambda x, y, y_: 3.6 * fps * abs(x) / (y_ - y)


def occupancy_estimation(motorbike, car, truck, area):
    return lambda x, y, z: (x * motorbike + y * car + z * truck) / area

def display(img, occupancy, motorbikes, small_car, big_car, avg_speed, area_cal, color=(0, 0, 0), isLeft = True, fontSize = 2):
    mask = np.ones((400, 450, 3))*255
    alpha = 0.5
    if isLeft:
      p = 0
      img[:400, :450] = (1-alpha)*img[:400, :450] + alpha*mask
    else:
      p = img.shape[1]-450
      img[:400, p:] = (1-alpha)*img[:400, p:] + alpha*mask

    cv2.putText(img, 'Average Speed: {:.2f}km/h'.format(avg_speed), (p, 50), 0, 1, color, fontSize)
    cv2.putText(img, 'Occupancy:{:.2f}%'.format(occupancy), (p, 100), 0, 1, color, fontSize)
    # cv2.putText(img, 'Stable: {:.2f}%'.format(stable), (p, 150), 0, 1, color, fontSize)
    cv2.putText(img, '# Motorbikes: %d' % motorbikes, (p, 150), 0, 1, color, fontSize)
    cv2.putText(img, '# Medium Vehicles: %d' % small_car, (p, 200), 0, 1, color, fontSize)
    cv2.putText(img, '# Large Vehicles: %d' % big_car, (p, 250), 0, 1, color, fontSize)
    cv2.putText(img, 'Area: %.2f m^2' % area_cal, (p, 300), 0, 1, color, fontSize)
    return img

def displayTotal(img, total_motorbike, total_small_car, total_big_car, color=(0, 0, 0), fontSize = 1):
    step = 30
    start = 0
    p = int(img.shape[1]/2) + 80
    cv2.putText(img, 'TOTAL VEHICLES', (p,start + step), 0, .5, (0, 0, 0), fontSize)
    cv2.putText(img, 'Motorbikes: %d' % total_motorbike, (p, start + 2*step), 0, .5, color, fontSize)
    cv2.putText(img, 'Medium Vehicles: %d' % total_small_car, (p, start + 3*step), 0, .5, color, fontSize)
    cv2.putText(img, 'Large Vehicles: %d' % total_big_car, (p, start + 4*step), 0, .5, color, fontSize)
    return img

def displayALL(img, occupancy, motorbikes, small_car, big_car, avg_speed, area_cal, total_motorbike, total_small_car, total_big_car,total_1_minutes, color=(0, 0, 0), fontSize = 1):
    fontScale  = .4
    step = 20
    start = 20
    p = 20
    cv2.putText(img, '                LEFT      RIGHT      TOTAL', (p,start+step), 0, .5, color, fontSize)
    cv2.putText(img, 'Average Speed:   {:.2f}       {:.2f} km/h'.format(avg_speed[0],avg_speed[1]), (p,start + 2*step), 0, fontScale, color, fontSize)
    cv2.putText(img, 'Occupancy:       {:.2f}       {:.2f} %'.format(occupancy[0],occupancy[1]), (p,start + 3*step), 0, fontScale, color, fontSize)
    # cv2.putText(img, 'Stable: {:.2f}%'.format(stable), (p, 150), 0, 1, color, fontSize)
    cv2.putText(img, '# Motorbikes:      %4d      %4d' % (motorbikes[0], motorbikes[1]),(p,start + 4*step), 0, fontScale, color, fontSize)
    cv2.putText(img, '# Medium Vehicles: %4d       %4d' % (small_car[0], small_car[1]), (p,start + 5*step), 0, fontScale, color, fontSize)
    cv2.putText(img, '# Large Vehicles:   %4d       %4d' % (big_car[0], big_car[1]), (p,start + 6*step), 0, fontScale, color, fontSize)
    cv2.putText(img, 'Area:             {:.2f}    {:.2f} m^2'.format(area_cal[0],area_cal[1]), (p,start + 7*step), 0, fontScale, color, fontSize)
    cv2.putText(img, 'Total Vehicles:      %4d       %4d        %4d' % (sum([total_motorbike[0],total_small_car[0],total_big_car[0]]), sum([total_motorbike[1],total_small_car[1],total_big_car[1]]),
    sum([sum(total_motorbike),sum(total_small_car),sum(total_big_car)])),(p,start + 8*step), 0, fontScale, color, fontSize)
    cv2.putText(img, '# Motorbikes:       %4d       %4d        %4d' % (total_motorbike[0], total_motorbike[1],sum(total_motorbike)),(p,start + 9*step), 0, fontScale, color, fontSize)
    cv2.putText(img, '# Medium Vehicles: %4d       %4d        %4d' % (total_small_car[0], total_small_car[1],sum(total_small_car)), (p,start + 10*step), 0, fontScale, color, fontSize)
    cv2.putText(img, '# Large Vehicles:   %4d       %4d        %4d' % (total_big_car[0], total_big_car[1],sum(total_big_car)), (p,start + 11*step), 0, fontScale, color, fontSize)
    cv2.putText(img, 'Vehicles/minutes:   %4d       %4d        %4d' % (total_1_minutes[0], total_1_minutes[1],sum(total_1_minutes)), (p,start + 12*step), 0, fontScale, color, fontSize)
    return img

def search_pivot(points):
    top, bottom, right, left = int(1e6), 0, 0, int(1e6)
    for points in points:
        if points[1] < top:
            top = points[1]
        elif points[1] > bottom:
            bottom = points[1]
        if points[0] < left:
            left = points[0]
        elif points[0] > right:
            right = points[0]
    return top, bottom, right, left


def median(lst):
  length = len(lst)
  lst.sort()
  if length % 2==0:
    return 0.5*(lst[length//2]+lst[length//2-1])
  else:
    return lst[length//2]


def loader(loader, vid):
  while vid.isOpened():
    res, org_img = vid.read()
    if not res:
      loader.put((False, False))
      break
    frame = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    loader.put((True, (org_img, frame)))


def writer(writer, out):
  while True:
    res, data = writer.read()
    if not res:
      break
    org_img, average_occupancy, motorbikes, small_car, big_car, average_speed, roi_points, stable, disp_speed = data
    display(org_img, 100 * average_occupancy, motorbikes, small_car, big_car, average_speed, roi_points, stable, disp_speed)
    out.write(org_img)


def get_pixel(rw, ppa, aw, delta, rh, vh, vw, diameter,  isLeft = False):
    # rw = 1.1*rw
    alpha = (180/math.pi)*math.acos(rh*math.tan(aw)/rw)
    if alpha < delta:
        dis = rh*math.tan(aw)/math.cos(delta*math.pi/180)
        percent_ship = (dis-rw)/dis
        x = vw -percent_ship*vw/2
        if isLeft:
           x = percent_ship*vw/2
        point = (x/vw, 1)
    else:
        px = (alpha-delta)*ppa
        x = vw
        if isLeft:
            x = 0
        point = (x/vw, (vh-px)/vh)

    top_alpha = math.atan(diameter/rh)*180/math.pi
    top_px = int((top_alpha-delta)*ppa)
    if top_px > vh:
        top_alpha = delta + vh/ppa
        top_px = vh
    distance =  rh*math.tan(aw)/math.cos(top_alpha*math.pi/180)
    ratio = (distance-rw)/distance
    x = vw -ratio*vw/2
    if isLeft:
        x = ratio*vw/2
    return (point, (x/vw, (vh-top_px)/vh))


def get_roi_points(touch_angle, h_angle, real_height, left_real_width, right_real_width, width, height, diameter, ppa):
    width_angle = h_angle*math.pi/180
    point_left = get_pixel(left_real_width, ppa, width_angle, touch_angle,real_height, height, width, diameter,  True)
    point_right = get_pixel(right_real_width, ppa, width_angle, touch_angle,real_height, height, width, diameter,  False)
    lst = [point_left[0], point_left[1], point_right[1]]
    t1 = point_left[0][1]
    t2 = point_right[0][1]
    if t1 == t2:
        lst.append(point_right[0])
        lst.append(point_left[0])
    else:
        if t2 > t1:
            rtl = (point_right[0], (point_left[0][0], t2))
        elif t1 > t2:
            rtl = (point_right[0], (point_right[0][0], t1))
            min = t2
        lst.append(rtl[0])
        lst.append(rtl[1])
        lst.append(point_left[0])
    bottom = touch_angle + (height-t1*height)/ppa
    top = touch_angle + (height-point_left[1][1]*height)/ppa
    diameter = math.tan((math.pi/180)*top)*real_height - math.tan((math.pi/180)*bottom)*real_height
    return lst, diameter


def read_ccfg(height, width, diameter):
  data = []

  if 'QT' in FLAGS.input:
    with open('./cfg/cfg.txt', 'r') as f:
      data = [float(i.strip()) for i in f.readlines()]

  elif 'SA' in FLAGS.input:
    with open('./cfg/cfg2.txt', 'r') as f:
      data = [float(i.strip()) for i in f.readlines()]

  else:
    raise FileNotFoundError('cfg')

  real_height = data[0]
  left_width = data[1]
  right_width = data[2]
  base_angle = data[3]
  delta_angle = data[4]
  h_angle = data[5]
  top_rank = data[6]
  ppa = 0.5*height/delta_angle
  app = 2*delta_angle/height
  touch_angle = abs(base_angle-delta_angle)
  roi_points, diameter = get_roi_points(touch_angle, h_angle, real_height, left_width, right_width, width, height, diameter, ppa)
  left_area = diameter*left_width
  right_area = diameter*right_width
  return lambda x: math.tan((touch_angle+app*(height-x))*math.pi/180)*real_height, roi_points, left_area, right_area, top_rank


def get_speeds(stage_speed, speeds):
    new_stage = median(speeds)
    if stage_speed > 30 and new_stage/stage_speed > 3:
      stage_speed = stage_speed
    else:
      stage_speed = new_stage
    return stage_speed


def polyroi(img, points, color=(0,0,0)):
    return cv2.fillPoly(img, points, color)

output_name = ''

class OUTPUT_TYPE(Enum):
    VIDEO_FOLDER = 1
    HISTORY_FOLDER = 2
    IMAGE_FOLDER = 3
    VIDEO = 4
    HISTORY_LEFT = 5
    HISTORY_RIGHT = 7
    IMAGE = 8


def get_output_path(output_type):
    global output_name
    if output_type == OUTPUT_TYPE.VIDEO_FOLDER:
        return os.path.join(FLAGS.output_folder, 'video')
    if output_type == OUTPUT_TYPE.HISTORY_FOLDER:
        return os.path.join(FLAGS.output_folder, 'history')
    if output_type == OUTPUT_TYPE.IMAGE_FOLDER:
        return os.path.join(FLAGS.output_folder, 'image')
    if output_type == OUTPUT_TYPE.VIDEO:
        return os.path.join(FLAGS.output_folder, 'video/' + output_name + '.avi')
    if output_type == OUTPUT_TYPE.HISTORY_LEFT:
        return os.path.join(FLAGS.output_folder, 'history/' + output_name + '_L_.csv')
    if output_type == OUTPUT_TYPE.HISTORY_RIGHT:
        return os.path.join(FLAGS.output_folder, 'history/' + output_name + '_R_.csv')
    if output_type == OUTPUT_TYPE.IMAGE:
        return os.path.join(FLAGS.output_folder, 'image/' + output_name)

def setup_output():
    os.makedirs(FLAGS.output_folder, exist_ok=True)
    os.makedirs(get_output_path(OUTPUT_TYPE.VIDEO_FOLDER), exist_ok=True)
    os.makedirs(get_output_path(OUTPUT_TYPE.HISTORY_FOLDER), exist_ok=True)
    os.makedirs(get_output_path(OUTPUT_TYPE.IMAGE_FOLDER), exist_ok=True)
    global output_name
    output_name = os.path.basename(FLAGS.input) + '_out0'
    k = 1
    while os.path.exists(get_output_path(OUTPUT_TYPE.VIDEO)):
        output_name = output_name[:-1] + str(k)
        k = k + 1

def main(_argv):
    if not os.path.isfile(FLAGS.input):
        print('Please check input file: ' + os.path.abspath(os.path.expanduser(os.path.expandvars(FLAGS.input))))
        return
    setup_output()
    input_size = FLAGS.size
    video_path = FLAGS.input

    vid = cv2.VideoCapture(video_path)
    out, width, height, fps = video_reader(vid, 3/4, 1/2)
    vid.set(cv2.CAP_PROP_POS_FRAMES, int(FLAGS.start_minutes * 60* fps)) # optional
    bkg_shape = (width//4, height//4)
    max_cosine_distance = 0.5
    diameter = 50

    nn_budget = None
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=32)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    od = ObjectDetection(FLAGS.weights, FLAGS.names, FLAGS.configs, input_size, FLAGS.confidence, FLAGS.threshold)

    distance_cal, roi_points, left_area, right_area, top_rank = read_ccfg(height, width, diameter)
    crowd_model = crowd_estimation(input_shape=(None, None, 3))
    roi_points, raw_rois = region_of_interested(roi_points, input_size, bkg_shape, width, height)
    max_point = max(map(lambda a:a[1],roi_points))
    min_point = min(map(lambda a:a[1],roi_points))
    center_point = int(max_point*.7)
    top, bottom, right, left = search_pivot(roi_points)

    crop_width = right-left
    crop_height = bottom-top
    crowd_mid = crop_width//16
    crowd_width = crop_width//8

    margin = np.asarray([left, top, left, top])
    base = np.asarray([crop_width, crop_height, crop_width, crop_height])

    roi = [[0,0]]
    roi.extend(roi_points)
    roi.extend([[0, height],[width,height],[width,0],[0,0]])
    poly_rois = np.asarray([roi], dtype=np.int32)
    crowd_estimation_time = 2 * fps // 4
    background_time = 3 * fps // 4
    history_left = []
    history_right = []
    history_count_object_up = []
    history_count_object_down = []
    track_id_counted = []
    frame_num = 0

    l_area_cal = occupancy_estimation(3, 15, 22.5, left_area)
    r_area_cal = occupancy_estimation(3, 15, 22.5, right_area)
    track_object = dict()

    speed_function = speed_estimation( fps=fps)
    frame_num = 0
    l_stage_speed = r_stage_speed = int(1e3)
    l_historical_speed = []
    r_historical_speed = []
    average_l_speed = average_r_speed = 0.0
    average_l_occupancy = average_r_occupancy = -0.001
    l_occupancies = []
    r_occupancies = []

    import pathlib
    with open(get_output_path(OUTPUT_TYPE.HISTORY_LEFT), 'w') as f:
      f.write("occupancy,motorbikes,small_car,big_car,avg_speed, total_motorbike,total_small_car,total_big_car,total_1_minutes")
    with open(get_output_path(OUTPUT_TYPE.HISTORY_RIGHT), 'w') as f:
      f.write("occupancy,motorbikes,small_car,big_car,avg_speed, total_motorbike,total_small_car,total_big_car,total_1_minutes")
    # bkg, s_bkg = background_execution(vid, bkg_shape, top, left, bottom, right, FLAGS.bg_init)

    # apply_bkg  = init_apply(bkg)
    # roi_mask = create_roi_mask((height,width), roi_points)
    total_l_motorbike, total_r_motorbike = 0, 0
    total_l_small_car, total_r_small_car = 0, 0
    total_l_big_car, total_r_big_car = 0, 0
    total_vehicles_l_1_minutes, total_vehicles_r_1_minutes = 0, 0
    total_l_temp, total_r_temp = {}, {}
    total_fps_count = fps*60
    for i in range(total_fps_count):
      total_l_temp[i] = 0
      total_r_temp[i] = 0

    while True:
        b = True
        res, org_img = vid.read()
        if not res:
          out.release()
          print('Video has ended or failed, try a different video format!')
          break
        frame = polyroi(cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)/255.,poly_rois)
        crop_frame = frame[top:bottom, left:right].copy()
        start_time = time.time()

        if frame_num % crowd_estimation_time == 0:
          y_predict = crowd_model.predict(np.expand_dims(crop_frame,0))[0]
          l_motorbikes = np.sum(y_predict[:, 0:crowd_mid])
          r_motorbikes = np.sum(y_predict[:, crowd_mid:crowd_width])
          b=False

        frame_num += 1
        if b:
          image_data = cv2.resize(crop_frame, (input_size, input_size))
          blob = np.expand_dims(np.transpose(image_data, (2,0,1)), 0)
          boxes, confidences, class_ids, idxs = od.predict(blob, base, np.asarray([0, 0, 0, 0]))
          l_speeds = []
          r_speeds = []
          l_small_car = r_small_car = l_big_car = r_big_car = 0
          if len(idxs) > 0:
            features = encoder(crop_frame, boxes)
            detections = [Detection(bbox, confidence, feature, class_id) for bbox, confidence, feature, class_id in zip(boxes, confidences, features, class_ids)]
            tracker.predict()
            tracker.update(detections)
            for track in tracker.tracks:
                bbox = (track.to_tlbr() + margin).astype(int)
                class_id = track.get_class_id()
                tp = bbox[1]
                # print(f'trachid: { track.track_id}, tp: {tp}, center: {center_point}, {bbox[1]},{bbox[3]}')
                if bbox[1] < center_point:
                  if track.track_id not in history_count_object_up:
                    history_count_object_up.append(track.track_id)
                  if track.track_id not in track_id_counted:
                    if track.track_id in history_count_object_down:
                      if class_id == 0:
                        if bbox[0] < width//2:
                          total_l_motorbike += 1
                        else:
                          total_r_motorbike += 1
                      if class_id == 1:
                        if bbox[0] < width//2:
                          total_l_small_car += 1
                        else:
                          total_r_small_car += 1
                      if class_id == 2:
                        if bbox[0] < width//2:
                          total_l_big_car += 1
                        else:
                          total_r_big_car += 1
                      track_id_counted.append(track.track_id)

                if bbox[3] > center_point:
                  if track.track_id not in history_count_object_down:
                    history_count_object_down.append(track.track_id)

                  if track.track_id not in track_id_counted:
                    if track.track_id in history_count_object_up:
                      if class_id == 0:
                        if bbox[0] < width//2:
                          total_l_motorbike += 1
                        else:
                          total_r_motorbike += 1
                      if class_id == 1:
                        if bbox[0] < width//2:
                          total_l_small_car += 1
                        else:
                          total_r_small_car += 1
                      if class_id == 2:
                        if bbox[0] < width//2:
                          total_l_big_car += 1
                        else:
                          total_r_big_car += 1
                      track_id_counted.append(track.track_id)

                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                if class_id == 0:
                  if track_object.get(track.track_id) is not None:
                      to = track_object[track.track_id]
                      object_speed = -1
                      if bbox[1] > top_rank:
                          his_fram = to[1]
                          pivot = bbox[1]
                          new_distance = to[0]
                          distance = distance_cal(pivot)
                          if to[0] != -1:
                            delta_distance = abs(distance-new_distance)
                            if delta_distance >= 4.0 or frame_num-his_fram > 2*fps:
                              object_speed = speed_function(delta_distance, to[1], frame_num)
                              his_fram = frame_num
                              new_distance = distance
                            else:
                              object_speed = to[2]
                          else:
                            new_distance = distance
                          track_object[track.track_id] = [new_distance, his_fram, object_speed]

                      if object_speed != -1:
                        if bbox[0] < width//2:
                          l_speeds.append(object_speed)
                        else:
                          r_speeds.append(object_speed)
                      org_img = cv2.putText(org_img,"%.2f" % (object_speed),
                              (int(0.5 * (bbox[0] + bbox[2]) - 6), int(0.5 * (bbox[1] + bbox[3]))), 2, 1, (255, 255, 255),
                              2)

                      # if track.track_id in track_id_counted:
                      #   org_img = cv2.putText(org_img, ".",
                      #       (int(0.5 * (bbox[0] + bbox[2]) - 10), int(0.5 * (bbox[1] + bbox[3]))), 2, 3, (0, 0, 255),
                      #       2)
                  else:
                      track_object[track.track_id] = [-1, frame_num, -1]
                if class_id == 1:
                    if bbox[0] < width//2:
                      l_small_car += 1
                    else:
                      r_small_car += 1
                elif class_id == 2:
                    if bbox[0] < width//2:
                      l_big_car += 1
                    else:
                      r_big_car += 1
                # draw_prediction(org_img, class_id,bbox[0],bbox[1],bbox[2],bbox[3])

          l_occupancies.append(l_area_cal(l_motorbikes, l_small_car, l_big_car))
          r_occupancies.append(r_area_cal(r_motorbikes, r_small_car, r_big_car))

          if len(l_speeds) > 0 :
              l_historical_speed.append(get_speeds(l_stage_speed, l_speeds))
          if len(r_speeds) > 0 :
              r_historical_speed.append(get_speeds(r_stage_speed, r_speeds))

          sum_l_temp =  sum([total_l_motorbike, total_l_small_car, total_l_big_car])
          sum_r_temp = sum([total_r_motorbike, total_r_small_car, total_r_big_car])
          total_vehicles_l_1_minutes = sum_l_temp - total_l_temp[frame_num%total_fps_count]
          total_vehicles_r_1_minutes = sum_r_temp - total_r_temp[frame_num%total_fps_count]
          total_l_temp[frame_num%total_fps_count] = sum_l_temp
          total_r_temp[frame_num%total_fps_count] = sum_r_temp

          if frame_num % fps == 0:
              t0 = median(l_historical_speed) if len(l_historical_speed) > 0 else average_l_speed
              if average_l_speed > 10 and t0 > 3*average_l_speed :
                t0 = average_l_speed
              average_l_speed = t0
              t1 = median(r_historical_speed) if len(r_historical_speed) > 0 else average_r_speed
              if average_r_speed > 10 and t1 > 3*average_r_speed:
                t1 = average_r_speed
              average_r_speed = t1
              average_l_occupancy = sum(l_occupancies)/fps
              average_r_occupancy = sum(r_occupancies)/fps
              l_occupancies = []
              l_historical_speed = []
              r_occupancies = []
              r_historical_speed = []

              history_left.append("%f,%f,%f,%f,%f,%f,%f,%f,%f" % (average_l_occupancy, l_motorbikes, l_small_car, l_big_car, average_l_speed, total_l_motorbike, total_l_small_car, total_l_big_car, total_vehicles_l_1_minutes))
              history_right.append("%f,%f,%f,%f,%f,%f,%f,%f,%f" % (average_r_occupancy, r_motorbikes, r_small_car, r_big_car, average_r_speed,total_r_motorbike, total_r_small_car, total_r_big_car, total_vehicles_r_1_minutes))

        now_fps = 1.0 / (time.time() - start_time)
        print("Time: %d #Frame: %d FPS:%.2f TL: %d" % (frame_num//fps, frame_num, now_fps, len(tracker.tracks)))
        if b:
          for index in range(1, len(roi_points)):
              org_img = cv2.line(org_img, roi_points[index - 1], roi_points[index], (0, 0, 255), 2)
          # contours = np.array([[0,height_count_object_up], [width,height_count_object_up], [width,height_count_object_down], [0,height_count_object_down]])
          # fill_poly_with_alpha(org_img, contours)
          # org_img = cv2.line(org_img, (0,height_count_object_up), (width,height_count_object_up), (0, 255, 0), 2)
          # org_img = cv2.line(org_img, (0,height_count_object_down), (width,height_count_object_down), (0, 255, 0), 2)
          org_img = cv2.line(org_img, (0,center_point), (width,center_point), (255, 0, 0), 2)
          # org_img = display(org_img, 100 * average_l_occupancy, l_motorbikes, l_small_car, l_big_car, average_l_speed, left_area, isLeft=True)
          # org_img = display(org_img, 100 * average_r_occupancy, r_motorbikes, r_small_car, r_big_car, average_r_speed, right_area, isLeft=False)
          # org_img = displayTotal(org_img,total_motorbike, total_small_car, total_big_car)
          org_img = cv2.resize(org_img, (width//2, height//2))
          # stable = cv2.cvtColor(cv2.resize(mask, (width//4, height//4)), cv2.COLOR_GRAY2RGB)
          # stable = apply_roi_mask(roi_mask, stable)
          density = cv2.resize(y_predict, (width//4, height//4))
          density = cv2.normalize(density, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
          density = cv2.applyColorMap(density, cv2.COLORMAP_JET)
          # t = cv2.vconcat([density, stable])
          blank_image = np.zeros((height//4, width//4,3), np.uint8)
          blank_image[:,:] =(255,255,255)
          # blank_image = displayTotal(blank_image,total_motorbike, total_small_car, total_big_car)
          blank_image = displayALL(blank_image, [100 * average_l_occupancy, 100 * average_r_occupancy],
          [l_motorbikes, r_motorbikes], [l_small_car, r_small_car], [l_big_car, r_big_car],
          [average_l_speed, average_r_speed], [left_area, right_area], [total_l_motorbike, total_r_motorbike], [total_l_small_car, total_r_small_car],
          [total_l_big_car, total_r_big_car], [total_vehicles_l_1_minutes, total_vehicles_r_1_minutes])
          # blank_image = displayOrg(blank_image, 100 * average_r_occupancy, r_motorbikes, r_small_car, r_big_car, average_r_speed, right_area)
          org_img = cv2.hconcat([org_img, cv2.vconcat([blank_image, density])])
          out.write(org_img)

          try:
            # cv.imwrite(f"E:/RS/frame{frame_num}.png",org_img)
            cv.imshow("OUT",org_img)
            cv.waitKey(1)
          except:
            pass



        if frame_num % fps == 0:
          with open(get_output_path(OUTPUT_TYPE.HISTORY_LEFT), 'a+') as f:
            f.write("\n"+"\n".join(history_left))
          with open(get_output_path(OUTPUT_TYPE.HISTORY_RIGHT), 'a+') as f:
            f.write("\n"+"\n".join(history_right))
          history_left = []
          history_right = []

        if frame_num == FLAGS.max_frame:
          out.release()
          return

        if frame_num == 10 or frame_num % 1000 == 0:
          cv.imwrite(os.path.join(get_output_path(OUTPUT_TYPE.IMAGE), f"{output_name}_{frame_num}.png") ,org_img)


def create_roi_mask(shape, points_array):
  mask = np.zeros(shape, dtype=np.uint8)
  white = (255,255,255)
  cv2.fillPoly(mask, np.int32([points_array]), white)
  return mask


def apply_roi_mask(roi_mask, image):
  roi = cv2.resize(src=roi_mask, dsize=(image.shape[1], image.shape[0]))
  image[np.where(roi==0)] = (0,0,0)
  return image


def fill_poly_with_alpha(img, contours, alpha = .3, color =(0,255,0)):
  overlay = img.copy()
  cv2.fillPoly(overlay, pts = [contours], color = color)
  cv2.addWeighted(overlay, alpha, img, 1 - alpha,0, img)
classes = ["2-w", "small", "large"]

COLORS = np.array([np.array([150,150.0,0]),np.array([0,150.0,150]),np.array([150,0,150.0])])

def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

if __name__ == '__main__':
    try:
      app.run(main)
    except SystemExit:
        pass
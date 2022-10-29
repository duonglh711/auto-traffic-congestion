import cv2
def video_reader(vid, out_width=None, out_height=None):
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    out_width = int(out_width * width) or width
    out_height = int(out_height * height) or height
    out = cv2.VideoWriter("E:/QT-iPhone-small.mp4", codec, fps, (out_width, out_height))
    return out, width, height, fps
vid = cv2.VideoCapture("E:/QT-iPhone.mp4")
out, width, height, fps = video_reader(vid, 3/4, 1/2)
frame_num = 0
while True:
    frame_num += 1
    res, org_img = vid.read()
    if not res or frame_num % 10000 == 0:
        break
    # cv2.imshow("out",org_img)
    # cv2.waitKey(1)
    out.write(org_img)
out.release()

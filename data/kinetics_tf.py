import tensorflow as tf
import cv2
import numpy as np


class KINETICS(object):

    def __init__(self, data_mode, every_ms=1000, seq_len=1, height=224, width=224):
        self.every_ms = every_ms
        self.seq_len = seq_len
        self.data_mode = data_mode
        self.height, self.width = height, width

    def read_one_video(self, filename, every_ms=1000):
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
            print >> sys.stderr, 'Error: Cannot open video file ' + filename
            return
        last_ts = -99999  # The timestamp of last retrieved frame.
        video_list = []
        while True:
            # Skip frames
            while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
                if not video_capture.read()[0]:
                return

            last_ts = video_capture.get(CAP_PROP_POS_MSEC)
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            frame = self.shortSide_resize(frame, 256)  # resize frame with short side = 256 pixels
            # frame = cv2.resize(frame, (256, 256))
            video_list.append(frame[:, :, ::-1])  # R,G,B 0-255 uint8

        return video_list

    def shortSide_resize(self, frame, shortSide=256):
        height, width, channel = frame.shape
        if width < height:
            ratio = shortSide / float(width)
        else:
            ratio = shortSide / float(height)
        frame = cv2.resize(frame, (int(ratio * width), int(ratio * height)))
        return frame

    def cv_random_crop(self, img, params=None):

        if params is None:
            height, width, _ = img.shape
            h, w = self.height, self.width
            # w = nprandom.uniform(0.6 * width, width)
            # h = nprandom.uniform(0.6 * height, height)
            left = nprandom.uniform(width - w)
            top = nprandom.uniform(height - h)
            # convert to integer rect x1,y1,x2,y2
            rect = np.array([int(left), int(top), int(left + w), int(top + h)])
            flip = random.random() < 0.5
        else:
            rect, flip = params

        img = img[rect[1]:rect[3], rect[0]:rect[2], :]

        return img, [rect, flip]

    def sample_one_frame(self, video_list):
        num_frame = len(video_list)
        frame_index = np.random.randint(0, num_frame)
        frame = video_list[frame_index]
        # random cropping
        frame = self.cv_random_crop(frame)

        return frame

    def sample_one_flow(self, video_list):
        optical_flow = cv2.DualTVL1OpticalFlow_create()
        num_frame = len(video_list)
        frame_index = np.random.randint(0, num_frame - 1)
        p_frame, n_frame = video_list[frame_index], video_list[frame_index + 1]  # R,G,B
        # Convert RGB to GRAY
        prev_frame = cv2.cvtColor(p_frame[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        cur_frame = cv2.cvtColor(n_frame[:, :, ::-1], cv2.COLOR_BGR2GRAY)
        # convert to gpu
        prev_frame_gpu = cv2.UMat(prev_frame)
        cur_frame_gpu = cv2.UMat(cur_frame)
        # Compute TVL1-flow
        flow_gpu = optical_flow.calc(prev_frame_gpu, cur_frame_gpu, None)
        # back to cpu
        flow = flow_gpu.get()
        # cropping
        flow = self.cv_random_crop(flow)
        return flow #(height, width, 2)

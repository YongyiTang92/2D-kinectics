"""Easily convert RGB video data (e.g. .avi) to the TensorFlow tfrecords file format with the provided 3 color channels.
 Allows to subsequently train a neural network in TensorFlow with the generated tfrecords.
 Due to common hardware/GPU RAM limitations, this implementation allows to limit the number of frames per
 video actually stored in the tfrecords. The code automatically chooses the frame step size such that there is
 an equal separation distribution of the video images. Implementation supports Optical Flow
 (currently OpenCV's calcOpticalFlowFarneback) as an additional 4th channel.
"""

from tensorflow.python.platform import gfile
from tensorflow.python.platform import flags
from tensorflow.python.platform import app
import cv2 as cv2
import numpy as np
import math
import os
import tensorflow as tf
import time
import json
import shutil
import librosa
import csv
import multiprocessing
import subprocess
import pandas as pd
from functools import partial

FLAGS = flags.FLAGS
flags.DEFINE_integer('n_videos_in_record', 5,
                     'Number of videos stored in one single tfrecord file')
flags.DEFINE_string('image_color_depth', "uint8",
                    'Color depth as string for the images stored in the tfrecord files. '
                    'Has to correspond to the source video color depth. '
                    'Specified as dtype (e.g. uint8 or uint16)')
flags.DEFINE_string('file_suffix', "*.mp4",
                    'defines the video file type, e.g. .mp4')
flags.DEFINE_string('video_source', './samples', 'Directory with video files')
flags.DEFINE_string('destination', './output_tmp/videos',
                    'Directory for storing tf records')
flags.DEFINE_string('jpg_path', './images_tmp', 'Directory with video files')
flags.DEFINE_string('json_path', './val.csv', 'Directory with json label files')
flags.DEFINE_integer('width_video', 320, 'the width of the videos in pixels')
flags.DEFINE_integer('height_video', 240, 'the height of the videos in pixels')
flags.DEFINE_integer('n_frames_per_video', 5,
                     'specifies the number of frames to be taken from each video')
flags.DEFINE_integer('FPS', 12,
                     'specifies the FPS to be taken from each video')
flags.DEFINE_integer('n_channels', 3,
                     'specifies the number of channels the videos have')
flags.DEFINE_string('video_filenames', None,
                    'specifies the video file names as a list in the case the video paths shall not be determined by the '
                    'script')
# flags.DEFINE_string('rtx_name', 'matthzhuang',
#                     'rtx_name for accessing hdfs')
# flags.DEFINE_string('proj_name', 'VideoAI',
#                     'proj_name for accessing hdfs')
# flags.DEFINE_string('token', 'e7412d14-f0e8-43a2-a84c-23fa418d5399',
#                     'token for accessing hdfs')
# flags.DEFINE_string('hdfs_dir', '/user/VideoAI/rextang/kinetics/videos/trimmed/tfrecords',
#                     'hdfs_dir for accessing hdfs')
flags.DEFINE_string('hdfs_dir', 'hdfs://100.77.5.27:9000/user/VideoAI/rextang/kinetics-400',
                    'hdfs_dir for accessing hdfs')
flags.DEFINE_integer('workers', 16,
                     'Number of workers for multiprocessing')
flags.DEFINE_integer('batch_start', 0,
                     'batch_start')
flags.DEFINE_integer('batch_end', 1,
                     'batch_end')


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def get_chunks(l, n):
    """Yield successive n-sized chunks from l.
    Used to create n sublists from a list l"""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_video_capture(path):
    assert os.path.isfile(path)
    cap = None
    if path:
        cap = cv2.VideoCapture(path)
    return cap


def get_next_frame(cap):
    ret, frame = cap.read()
    if not ret:
        return None

    return np.asarray(frame)


def mp4_to_jpgs(video_name, jpg_path, fps, filenames):
    # filenames = os.path.join(FLAGS.video_source, video_name + '.mp4')
    # assert os.path.isfile(filenames)
    image_path = os.path.join(jpg_path, video_name)
    if not os.path.isdir(image_path):
        os.makedirs(image_path)
    decode_command = 'ffmpeg -i ' + filenames + ' -r ' + str(fps) + ' ' + image_path + '/frame%05d.jpg'
    result = subprocess.Popen(decode_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, close_fds=True)
    output, error = result.communicate()
    assert error[-8:-2] != 'failed'


def clear_path(jpg_path, video_name):
    image_path = os.path.join(jpg_path, video_name)
    shutil.rmtree(image_path, ignore_errors=False, onerror=None)


def upload_tfrecords(tfrecord_dir, tfrecord_base):
    # tfrecord_dir_video = os.path.join(tfrecord_dir, video_name)
    # tfrecord_list = gfile.Glob(os.path.join(tfrecord_dir_video, '*.tfrecords'))
    # if subset == 'train':
    #     subset_name = 'training'
    # elif subset == 'val':
    #     subset_name = 'validation'
    # elif subset == 'test':
    #     subset_name = 'test'
    # else:
    #     raise('subset_name error: ', subset_name)
    # for i, file in enumerate(tfrecord_list):
    #     print(file)
    #     upload_command = 's_fs -u ' + FLAGS.rtx_name + ' -b ' + FLAGS.proj_name + ' -t ' + FLAGS.token + ' -put ' + file + ' ' + FLAGS.hdfs_dir + '/' + subset_name
    #     # print(upload_command)
    #     result = os.system(upload_command)

    upload_command = 'hdfs dfs -put ' + tfrecord_dir + ' ' + FLAGS.hdfs_dir + '/' + tfrecord_base
    # print(upload_command)
    result = os.system(upload_command)
    os.remove(tfrecord_dir)


def decode_audio(file_path, sample_per_sec=22050):
    audio, _ = librosa.load(file_path, sr=sample_per_sec, mono=False)  # Extract audio
    pro_audio = preprocess_audio(audio, sample_per_sec)
    return pro_audio


# def get_hdfs_files(hdfs_dir):
#     check_command = 's_fs -u $RTX_NAME -b $BUS_NAME -t $TOKEN -ls -R ' + hdfs_dir
#     r = os.popen(check_command)
#     files_list = []
#     for line in r:
#         line = line.strip('\r\n')
#         file_path = line.split(' ')[-1]
#         if '.tfrecords' in file_path:
#             video_name = file_path.split('/')[-1][:11]
#             if video_name not in files_list:
#                 files_list.append(video_name)

#     return files_list


def preprocess_audio(raw_audio, sample_per_sec, minimum_seconds=10):
    # Re-scale audio from [-1.0, 1.0] to [-256.0, 256.0]
    # Return audio with size max(sample_per_sec*minimum_seconds, sample_per_sec*ground_truth_second)
    # Select first channel (mono)
    if len(raw_audio.shape) > 1:
        raw_audio = raw_audio[0]

    raw_audio[raw_audio < -1.0] = -1.0
    raw_audio[raw_audio > 1.0] = 1.0

    # Make range [-256, 256]
    raw_audio *= 256.0

    # Make minimum length available
    min_length = sample_per_sec * minimum_seconds
    if min_length > raw_audio.shape[0]:
        raw_audio = np.tile(raw_audio, int(min_length / raw_audio.shape[0]) + 1)

    # Check conditions
    assert len(raw_audio.shape) == 1, "It seems this audio contains two channels, we only need the first channel"
    assert np.max(raw_audio) <= 256, "It seems this audio contains signal that exceeds 256"
    assert np.min(raw_audio) >= -256, "It seems this audio contains signal that exceeds -256"

    # Shape to 1 x DIM x 1 x 1
    # raw_audio = np.reshape(raw_audio, [1, -1, 1, 1])

    return raw_audio.copy()


def extract_flow(images_name):
    # Extract TVL1 flow and encode it as jpg
    num_frames = len(images_name)
    optical_flow = cv2.DualTVL1OpticalFlow_create()

    flow_x_list = []
    flow_y_list = []
    for i_frames in range(num_frames):
        prev_frame = cv2.imread(images_name[max(0, i_frames - 1)])
        cur_frame = cv2.imread(images_name[i_frames])
        prev_frame = img_resize_256(prev_frame)
        cur_frame = img_resize_256(cur_frame)
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2GRAY)
        # Compute TVL1-flow
        prev_frame_gpu = cv2.UMat(prev_frame)
        cur_frame_gpu = cv2.UMat(cur_frame)
        flow_gpu = optical_flow.calc(prev_frame_gpu, cur_frame_gpu, None)
        flow = flow_gpu.get()
        # truncate [-20, 20]
        flow[flow >= 20] = 20
        flow[flow <= -20] = -20
        # scale to [0, 255]
        flow = flow + 20
        flow = flow / 40  # normalize the data to 0 - 1
        flow = 255 * flow  # Now scale by 255
        flow = flow.astype(np.uint8)
        # Encode to jpg
        flow_x_encode = cv2.imencode('.jpg', flow[:, :, 0])[1]
        flow_y_encode = cv2.imencode('.jpg', flow[:, :, 1])[1]
        flow_x_encode = flow_x_encode.tostring()
        flow_y_encode = flow_y_encode.tostring()
        flow_x_list.append(_bytes_feature(flow_x_encode))
        flow_y_list.append(_bytes_feature(flow_y_encode))

    return flow_x_list, flow_y_list


def img_resize_256(img):
    h, w = img.shape[0], img.shape[1]
    if h <= w:
        h_pi = int((h * 256.0) / h)
        w_pi = int((w * 256.0) / h)
    else:
        h_pi = int((h * 256.0) / w)
        w_pi = int((w * 256.0) / w)
    img_resized = cv2.resize(img, (w_pi, h_pi))
    return img_resized


def resize_img_list(images_name):
    num_frames = len(images_name)
    resize_image_list = []
    for i_frames in range(num_frames):
        frame = cv2.imread(images_name[i_frames])
        frame = img_resize_256(frame)
        image_encode = cv2.imencode('.jpg', frame)[1]
        image_encode = image_encode.tostring()
        resize_image_list.append(_bytes_feature(image_encode))
    return resize_image_list


def convert_videos_to_tfrecord(source_path, destination_path, jpg_path, json_path,
                               n_videos_in_record=10, n_frames_per_video=5,
                               file_suffix="*.mp4", fps=12,
                               n_channels=3, width=1280, height=720,
                               color_depth="uint8", video_filenames=None):
    """calls sub-functions convert_video_to_numpy and save_numpy_to_tfrecords in order to directly export tfrecords files
    Args:
        source_path: directory where video videos are stored
        destination_path: directory where tfrecords should be stored
        n_videos_in_record: Number of videos stored in one single tfrecord file
        n_frames_per_video: specifies the number of frames to be taken from each
            video
        file_suffix: defines the video file type, e.g. *.mp4
            dense_optical_flow: boolean flag that controls if optical flow should be
            used and added to tfrecords
        n_channels: specifies the number of channels the videos have
        width: the width of the videos in pixels
        height: the height of the videos in pixels
        color_depth: Color depth as string for the images stored in the tfrecord
        files. Has to correspond to the source video color depth. Specified as
            dtype (e.g. uint8 or uint16)
        video_filenames: specify, if the the full paths to the videos can be
            directly be provided. In this case, the source will be ignored.
    """
    # activity_index, csv_data = _import_ground_truth(json_path)
    activity_index, _ = _import_ground_truth('/data1/rextang/kinetics_400/v1-0/train.csv')
    csv_data = pd.read_csv(json_path)
    data_split = json_path.split('/')[-1].split('.')[0]
    if not activity_index:
        raise RuntimeError('No activity_index files found.')
    if not os.path.isdir(destination_path):
        os.makedirs(destination_path)

    database = csv_data
    video_name_list = database['youtube_id']
    # batch_size = FLAGS.workers * 5
    batch_size = 100
    # existing_files = get_hdfs_files(FLAGS.hdfs_dir)
    total_video = len(video_name_list)
    st = time.time()
    for i in range(FLAGS.batch_start, min(total_video // batch_size + 1, FLAGS.batch_end)):
        # for i in range(FLAGS.batch_start, FLAGS.batch_end):
        print('Processing {:04d} of {:04d} batches, time/batch: {:.4f}s'.format(i + 1, min(total_video // batch_size + 1,
                                                                                           FLAGS.batch_end), time.time() - st))
        st = time.time()
        processing_tfrecord_upload(video_name_list[i * batch_size: min(len(database), (i + 1) * batch_size)],
                                   index=i,
                                   destination_path=destination_path, database=database, data_split=data_split,
                                   jpg_path=jpg_path, fps=fps, activity_index=activity_index)

        # pool = multiprocessing.Pool(processes=FLAGS.workers)
        # pool.map(partial(processing_tfrecord_upload, destination_path=destination_path, database=database,
        #                  jpg_path=jpg_path, fps=fps, activity_index=activity_index, existing_files=existing_files), video_name_list[i * batch_size: min(len(database), (i + 1) * batch_size)])
        # pool.close()
        # pool.join()


def processing_tfrecord_upload(_video_name_list, index, destination_path, database, data_split, jpg_path, fps, activity_index):
    print('current index is %d' % index)

    if not os.path.isdir(os.path.join(destination_path, data_split)):
        os.makedirs(os.path.join(destination_path, data_split))
    tfrecord_base = os.path.join(data_split, (data_split + '{:04d}.tfrecord'.format(index)))
    tfrecord_dir = os.path.join(destination_path, tfrecord_base)
    writer = tf.python_io.TFRecordWriter(tfrecord_dir)

    for _video_name in _video_name_list:
        frame_data = database[database['youtube_id'] == _video_name]  # CSV data for the current video
        # decode mp42jpgs with ffmpeg
        video_name = _video_name + '_{:06d}_{:06d}'.format(int(frame_data['time_start'].item()),
                                                           int(frame_data['time_end'].item()))
        mp4_filenames = os.path.join(FLAGS.video_source, video_name + '.mp4')
        try:

            assert os.path.isfile(mp4_filenames)
            # Decode video to mp4
            mp4_to_jpgs(video_name, jpg_path, fps, mp4_filenames)
            example = save_jpgs_to_tfrecords(video_name, frame_data, jpg_path,
                                             fps, activity_index, mp4_filenames)

            # delete jpgs file
            clear_path(jpg_path, video_name)
            # upload tfrecord according to different subset
            writer.write(example.SerializeToString())
            print('Video %s succeed' % _video_name)

        except:
            print('Video %s failed' % _video_name)
            with open('failed_videos_' + data_split + '.csv', 'a') as myfile:
                csv_writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                csv_writer.writerow([video_name])

    writer.close()
    # save tf record and upload
    upload_tfrecords(tfrecord_dir, tfrecord_base)


def save_jpgs_to_tfrecords(video_name, frame_data, jpg_path, fps, activity_index, mp4_filenames, audio_fps=22050):
    """Converts an entire dataset into x tfrecords where x=videos/fragmentSize.
    Args:
        data: ndarray(uint32) of shape (v,i,h,w,c) with v=number of videos,
        i=number of images, c=number of image channels, h=image height, w=image
        width
        name: filename; data samples type (train|valid|test)
        total_batch_number: indicates the total number of batches
    """
    # if not os.path.isdir(os.path.join(destination_path, video_name)):
    #     os.makedirs(os.path.join(destination_path, video_name))

    images_name = gfile.Glob(os.path.join(jpg_path, video_name, '*.jpg'))  # Read all jpg-name
    images_name.sort()
    try:
        audio = decode_audio(mp4_filenames, audio_fps)
        current_audio_raw = audio.tostring()
    except:
        audio_len = (len(images_name) // fps) * audio_fps
        audio = np.asarray(np.random.randint(-256, 256, audio_len), np.float32)
        current_audio_raw = audio.tostring()

    try:
        info_label = frame_data['label'].item()
    except:
        info_label = 'unknown'
    # image_list = [tf.gfile.FastGFile(images_name[image_count], 'rb').read() for image_count in range(len(images_name))]
    image_list = resize_img_list(images_name)
    number_of_frames = len(images_name)
    flow_x_list, flow_y_list = extract_flow(images_name)
    assert len(flow_x_list) == len(image_list), "The number of RGB not equal to the number of flow"
    feature_list = {}
    feature_list['images'] = tf.train.FeatureList(feature=image_list)
    feature_list['flow_x'] = tf.train.FeatureList(feature=flow_x_list)
    feature_list['flow_y'] = tf.train.FeatureList(feature=flow_y_list)
    context_features = {}
    context_features['audio'] = _bytes_feature(current_audio_raw)
    context_features['number_of_frames'] = _int64_feature(number_of_frames)
    context_features['video'] = _bytes_feature(str.encode(video_name))
    context_features['label_index'] = _int64_feature(activity_index[info_label])
    context_features['label_name'] = _bytes_feature(str.encode(info_label))

    example = tf.train.SequenceExample(
        context=tf.train.Features(feature=context_features),
        feature_lists=tf.train.FeatureLists(feature_list=feature_list))
    # writer.write(example.SerializeToString())
    # writer.close()
    return example


def _import_ground_truth(ground_truth_filename):
    """Reads ground truth file, checks if it is well formatted, and returns
       the ground truth instances and the activity classes.
    Parameters
    ----------
    ground_truth_filename : str
        Full path to the ground truth json file.
    Outputs
    -------
    activity_index : dict
        Dictionary containing class index.
    """

    data = pd.read_csv(ground_truth_filename)

    # Initialize data frame
    activity_index, cidx = {}, 0
    for videoid, label in data['label'].items():
        if label not in activity_index:
            activity_index[label] = cidx
            cidx += 1
    activity_index['unknown'] = cidx
    return activity_index, data


def main(argv):
    convert_videos_to_tfrecord(FLAGS.video_source, FLAGS.destination, FLAGS.jpg_path, FLAGS.json_path,
                               FLAGS.n_videos_in_record,
                               FLAGS.n_frames_per_video, FLAGS.file_suffix,
                               FLAGS.FPS, FLAGS.n_channels,
                               FLAGS.width_video, FLAGS.height_video,
                               FLAGS.image_color_depth, FLAGS.video_filenames)


if __name__ == '__main__':
    app.run()

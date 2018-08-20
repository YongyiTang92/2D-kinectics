import tensorflow as tf


class KineticsReader(BaseReader):
    """Reads TFRecords of SequenceExamples.

    The TFRecords must contain SequenceExamples with the sparse in64 'labels'
    context feature and a fixed length byte-quantized feature vector, obtained
    from the features in 'feature_names'. The quantized features will be mapped
    back into a range between min_quantized_value and max_quantized_value.
    """

    def __init__(self,
                 num_classes=400,
                 data_type='rgb'):
        """Construct a YT8MFrameFeatureReader.

        Args:
          num_classes: a positive integer for the number of classes.
          feature_sizes: positive integer(s) for the feature dimensions as a list.
          feature_names: the feature name(s) in the tensorflow record as a list.
          max_frames: the maximum number of frames to process.
        """

        self.num_classes = num_classes
        self.data_type = data_type
        if self.data_type == 'rgb':
            self.feature_names = ['images']
        elif self.data_type == 'flow':
            self.feature_names = ['flow_x', 'flow_y']
        else:
            raise 'Undefined data_type %s ' % self.data_type


def parse_exmp(serial_exmp):
    contexts, features = tf.parse_single_sequence_example(
        serialized_example,
        context_features={"id": tf.FixedLenFeature(
            [], tf.string),
            "labels": tf.VarLenFeature(tf.int64)},
        sequence_features={
            feature_name: tf.FixedLenSequenceFeature([], dtype=tf.string)
            for feature_name in self.feature_names
        })
    feats = tf.parse_single_example(serial_exmp, features={'feature': tf.FixedLenFeature([], tf.string),
                                                           'label': tf.FixedLenFeature([10], tf.float32), 'shape': tf.FixedLenFeature([], tf.int64)})
    image = tf.decode_raw(feats['feature'], tf.float32)
    label = feats['label']
    shape = tf.cast(feats['shape'], tf.int32)
    return image, label, shape

import tensorflow as tf
import os
def serialize_example(input_tensor, label_tensor):
    feature = {
        'input': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(input_tensor).numpy()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(label_tensor).numpy()]))
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()
def write_tfrecord(file_path, dataset):
    with tf.io.TFRecordWriter(file_path) as writer:
        for input_tensor, label_tensor in dataset:
            example = serialize_example(input_tensor, label_tensor)
            writer.write(example)
def parse_example(example_proto):
    feature_description = {
        'input': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example_proto, feature_description)
    input_tensor = tf.io.parse_tensor(example['input'], out_type=tf.float32)
    label_tensor = tf.io.parse_tensor(example['label'], out_type=tf.float32)
    return input_tensor, label_tensor
def read_tfrecord(file_path):
    raw_dataset = tf.data.TFRecordDataset(file_path)
    parsed_dataset = raw_dataset.map(parse_example)
    return parsed_dataset



def load_data(data_dir, data_name,train_size, batch_size):
    """
    Collects the tf recod files that make up a dataset and returns the file
    lists for training, validation and testset

    Parameters
    ----------
    data_dir : str
        Path of the folder containing the tf record files
    data_name : str
        Identifyer of the dataset

    Returns
    -------
    train_files : list of str
        List with the filepaths of the tf record files in the training split
    val_files : list of str
        List with the filepaths of the tf record files in the validation split
    test_files : list of str
        List with the filepaths of the tf record files in the test split

    """
    files = os.listdir(data_dir)
    train_path = [os.path.join(data_dir, f) for f in files
                               if f.startswith(data_name + '_') and
                                                  '_train_' in f and '.tfrecord' in f]
    val_path = [os.path.join(data_dir, f) for f in files
                                 if f.startswith(data_name + '_') and
                                                  '_val_' in f and '.tfrecord' in f]
    test_path = [os.path.join(data_dir, f) for f in files
                                      if f.startswith(data_name + '_') and
                                                        '_test_' in f and '.tfrecord' in f]
    train_dataset = read_tfrecord(train_path)
    val_dataset = read_tfrecord(val_path)
    test_dataset = read_tfrecord(test_path)

    train_set = train_dataset.shuffle(train_size).batch(batch_size, drop_remainder=True)
    val_set = val_dataset.batch(batch_size, drop_remainder=True)

    test_set = test_dataset.batch(batch_size, drop_remainder=True)

    return train_set, val_set, test_set

def file_starts_with(folder_path, starts_with_string):
  for filename in os.listdir(folder_path):
    if filename.startswith(starts_with_string):
      return True
  return False

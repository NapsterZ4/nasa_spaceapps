import time
import os
import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

os_path = os.getcwd()

PATH_TO_CFG = os_path + '/workspace/training_demo/exported-models/my_model/pipeline.config'
PATH_TO_CKPT = os_path + '/workspace/training_demo/exported-models/my_model/checkpoint'
PATH_TO_LABELS = os_path + '/workspace/training_demo/annotations/label_map.pbtxt'
IMAGE_PATH = '/mnt/napster_disk/space_apps/nasa_deploy/workspace/training_demo/images/fralome_05_delay-0.33s.jpg'


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def load_image_into_array(path):
    return np.array(Image.open(path))


print('Loading model... ', end='')
start_time = time.time()

# Load pipeline and build a detection model
configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
model_config = configs['model']
detection_model = model_builder.build(model_config=model_config, is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(PATH_TO_CKPT, 'ckpt-0')).expect_partial()

end_time = time.time()
elapsed_time = end_time - start_time
print('Done charge model, took {} seconds'.format(round(elapsed_time)))

# Load label map data (for plotting)
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

print("Running inference for image {}...".format(IMAGE_PATH), end='')
image_np = load_image_into_array(IMAGE_PATH)
input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
detections = detect_fn(input_tensor)

num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
              for key, value in detections.items()}
detections['num_detections'] = num_detections
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

label_id_offset = 1
image_np_with_detections = image_np.copy()

viz_utils.visualize_boxes_and_labels_on_image_array(
    image_np_with_detections,
    detections['detection_boxes'],
    detections['detection_classes'] + label_id_offset,
    detections['detection_scores'],
    category_index,
    use_normalized_coordinates=True,
    max_boxes_to_draw=200,
    min_score_thresh=.10,
    agnostic_mode=False)

plt.figure()
save = Image.fromarray(image_np_with_detections)
save.save("prueba2.jpg")
print('\nDone')

import time
import os
import tensorflow as tf
import streamlit as st
import numpy as np
from PIL import Image
import warnings

warnings.filterwarnings('ignore')
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

os_path = os.getcwd()
print(os_path)

PATH_TO_CFG = os_path + '/workspace/training_demo/exported-models/my_model/pipeline.config'
PATH_TO_CKPT = os_path + '/workspace/training_demo/exported-models/my_model/checkpoint'
PATH_TO_LABELS = os_path + '/workspace/training_demo/annotations/label_map.pbtxt'


def load_image_into_array(path):
    return np.array(Image.open(path))


@st.cache(allow_output_mutation=True)
def detection_hurricane(IMAGE_PATH):
    print('Loading model... ', end='')
    start_time = time.time()

    # Load pipeline and build a detection model
    configs = config_util.get_configs_from_pipeline_file(PATH_TO_CFG)
    model_config = configs['model']
    detection_model = model_builder.build(model_config=model_config, is_training=False)

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

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
        min_score_thresh=.05,
        agnostic_mode=False)

    save = Image.fromarray(image_np_with_detections)
    return save


st.set_option('deprecation.showfileUploaderEncoding', False)


def streamlit_controller():
    # image = Image.open("Insert image logotype")
    st.title("PROTOTIPO DE DETECCIÓN Y PREDICCIÓN DE HURACANES A PARTIR DE IMÁGENES SATELITALES DE LA NASA")
    # st.image(image, use_column_width=True)

    uploaded_file = st.file_uploader("Insert Image to predict", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image_up = Image.open(uploaded_file)
        image_down = load_image_into_array(uploaded_file)
        array_image = Image.fromarray(image_down)
        array_image.save(os_path + "uploaded_image.png")

        image_upload = os_path + "uploaded_image.png"
        st.image(image_up, use_column_width=True)
        save_image = detection_hurricane(image_upload)

        save_image.save(os_path + "/test_image.png")
        processing_image = os_path + "/test_image.png"
        st.success("DETECTION HURRICANE")
        st.image(processing_image, use_column_width=True)
        st.success("Done")
    else:
        st.warning("Upload file")


if __name__ == '__main__':
    streamlit_controller()
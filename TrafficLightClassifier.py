import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import argparse

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
ap.add_argument("-m", "--frozen-model-path", required=False,
                default='./object_detection/fine_tuned_models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb',
                help="path to the frozen traffic light detection model")
args = vars(ap.parse_args())

image_path = args['image']

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './object_detection/config/labels_map.pbtxt'

NUM_CLASSES = 4

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(args['frozen_model_path'], 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)

category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


image = Image.open(image_path)
image_np = load_image_into_numpy_array(image)

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: np.expand_dims(image_np, axis=0)})

        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=2)

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
        while True:
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

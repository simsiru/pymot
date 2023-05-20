"""Multi object tracking module"""

import cv2
import numpy as np
from pymot.tracking.deep_sort import nn_matching
from pymot.tracking.deep_sort.detection import Detection
from pymot.tracking.deep_sort.tracker import Tracker
from pymot.tracking.deep_sort.pytorch_reid_feature_extractor import (
    Extractor,
    get_features,
)
from pymot.object_detection import yolo
from pymot.utils import draw_bbox_tracking


class MOT:
    """Multi object tracker class.

    Parameters
    ----------
    mot_cfg : dict
        Multi object tracker configuration (template below).

        mot_cfg = {
           'od_classes':'SPECIFY', # Object detection algorithm classes path
           'od_algo':'yolo',
           'od_wpath':'SPECIFY',   # Object detection algorithm weights path
           'od_cpath':'SPECIFY',   # Object detection algorithm config path
           'od_nms_thr':0.4,
           'od_conf_thr':0.5,
           'od_img_size':416,
           'od_cuda':True,
           't_classes':'SPECIFY',  # List of classes to track
           't_algo':'deepsort',
           't_cuda':True,
           't_metric':'cosine',
           't_max_cosine_distance':0.2,
           't_budget':100,
           't_max_iou_distance':0.7,
           't_max_age':70,
           't_n_init':3
        }
    """

    def __init__(self, mot_cfg: dict) -> None:
        with open(mot_cfg["od_classes"]) as f:
            self.od_classes = f.read().split('\n')

        self.track_classes = mot_cfg["t_classes"]

        if mot_cfg["od_algo"] == "yolo":
            self.od_model = yolo.YOLO(
                mot_cfg["od_wpath"],
                mot_cfg["od_cpath"],
                nms_thr=mot_cfg["od_nms_thr"],
                conf_thr=mot_cfg["od_conf_thr"],
                img_size=mot_cfg["od_img_size"],
                enable_cuda=mot_cfg["od_cuda"],
            )

        if mot_cfg["t_algo"] == "deepsort":
            self.feature_extractor = Extractor(use_cuda=mot_cfg["t_cuda"])

            metric = nn_matching.NearestNeighborDistanceMetric(
                mot_cfg["t_metric"],
                mot_cfg["t_max_cosine_distance"],
                budget=mot_cfg["t_budget"],
            )

            self.tracker = Tracker(
                metric,
                max_iou_distance=mot_cfg["t_max_iou_distance"],
                max_age=mot_cfg["t_max_age"],
                n_init=mot_cfg["t_n_init"],
            )

    def detect_objects(
        self, frame: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect objects in a frame using an object detection algorithm.

        Parameters
        ----------
        frame : ndarray
            RGB frame as a numpy array.

        Returns
        -------
        (ndarray, ndarray, ndarray)
            Returns a tuple of bounding boxes, scores and class names.
        """

        boxes, scores, names = [], [], []

        (class_ids, probs, bboxes) = self.od_model.detect(frame)

        for i, box in enumerate(bboxes):
            if self.od_classes[class_ids[i]] in self.track_classes:
                boxes.append([box[0], box[1], box[2], box[3]])
                scores.append(probs[i])
                names.append(self.od_classes[class_ids[i]])

        boxes = np.array(boxes)
        scores = np.array(scores)
        names = np.array(names)

        return (boxes, scores, names)

    def track_objects(
        self,
        frame: np.ndarray,
        processing_func=None,
        **proc_kwargs
    ) -> tuple[dict, np.ndarray, dict]:
        """Detect objects in a frame using an object detection algorithm.

        Parameters
        ----------
        frame : ndarray
            RGB frame as a numpy array.

        Returns
        -------
        (dict, ndarray, dict)
            Returns a tuple of information dict of current objects,
            frame with current tracked objects and, if processing function
            is supplied, the information dict of processed tracked objects.
        """

        height, width = frame.shape[0], frame.shape[1]

        frame_with_bboxes = frame.copy()

        # Obtain all the detections for the given frame.
        boxes, scores, names = self.detect_objects(frame)

        features = get_features(self.feature_extractor, boxes, frame)

        detections = [
            Detection(bbox, score, class_name, feature)
            for bbox, score, class_name, feature in zip(
                boxes, scores, names, features
            )
        ]

        # Pass detections to the deepsort object and obtain
        # the track information.
        self.tracker.predict()
        self.tracker.update(detections)

        n_objects = len(self.tracker.tracks)
        current_obj = {}
        proc_obj_info = (None, None, None)

        # Obtain info from the tracks
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 5:
                continue

            # Get the corrected/predicted bounding box
            bbox = track.to_tlbr()

            # Get the class name of particular object
            class_name = track.get_class()

            # Get the ID for the particular track
            tracking_id = track.track_id

            xmin, ymin, xmax, ymax = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            )

            if xmin < 0:
                xmin = 0
            if ymin < 0:
                ymin = 0
            if xmax > width:
                xmax = width
            if ymax > height:
                ymax = height

            obj_coord = (xmin, ymin, xmax, ymax)

            current_obj[tracking_id] = (class_name, obj_coord)

            if processing_func is not None:
                proc_obj_info, viz_info = processing_func(
                    frame, tracking_id, obj_coord, **proc_kwargs
                )

                draw_bbox_tracking(
                    obj_coord,
                    height,
                    frame_with_bboxes,
                    tracking_id,
                    class_name,
                    rand_colors=viz_info[0],
                    rec_bool=viz_info[1],
                    colors=viz_info[2],
                    info_text=proc_obj_info[0][tracking_id][2],
                    unknown_obj_info=viz_info[3],
                )
            else:
                draw_bbox_tracking(
                    obj_coord,
                    height,
                    frame_with_bboxes,
                    tracking_id,
                    class_name,
                    rand_colors=True,
                    rec_bool=False,
                    colors=((0, 255, 0), (0, 0, 255)),
                    info_text="",
                    unknown_obj_info="UNKNOWN",
                )

        cv2.rectangle(
            frame_with_bboxes,
            (0, 46),
            (280, 0),
            (255, 255, 255),
            thickness=cv2.FILLED,
        )

        cv2.putText(
            frame_with_bboxes,
            "Number of objects: {}".format(n_objects),
            (0, 40),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

        return (current_obj, frame_with_bboxes, proc_obj_info)

# Multi object tracking class

> Perform multi object tracking and use a custom function for tracked objects

MOT arguments:

-   _mot_cfg_ - configuration dictionary for MOT (tamplate below)

```python
mot_cfg = {
    'od_classes':'SPECIFY',       # Object detection algorithm classes path
    'od_algo':'yolo',
    'od_wpath':'SPECIFY',         # Object detection algorithm weights path
    'od_cpath':'SPECIFY',         # Object detection algorithm config path
    'od_nms_thr':0.4,
    'od_conf_thr':0.5,
    'od_img_size':416,
    'od_cuda':True,
    't_classes':'SPECIFY',        # List of classes to track
    't_algo':'deepsort',
    't_cuda':True,
    't_metric':'cosine',
    't_max_cosine_distance':0.2,
    't_budget':100,
    't_max_iou_distance':0.7,
    't_max_age':70,
    't_n_init':3
}
```

MOT methods:

- _detect_objects_ - detect object from a frame
- _track_objects_ - perform a tracking update on frame

## Algorithms

Currently available object detection algorithms:
        
- YOLOv3
- YOLOv4
        
Currently available tracking algorithms:
        
- DeepSORT

## Installation

To use the package torch, torchvision and opencv must be also installed.

Note: It is recommended to use opencv compiled with CUDA support, because algorithms that use opencv's darknet
backend like YOLOv3/v4 run an order of magnitude faster on a GPU.

```sh
pip install pymot
```

## Usage

Without a custom function for tracked objects

```python
from pymot.mot import MOT
import cv2

mot_tracker = MOT(mot_cfg)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break

    tracked_ojb_info, frame_with_bboxes, _ = mot_tracker.track_objects(frame)

    cv2.imshow('Object tracking with deepSORT', frame_with_bboxes)
```

With a custom function for tracked objects

```python
from pymot.mot import MOT
import cv2

mot_tracker = MOT(mot_cfg)

cap = cv2.VideoCapture(0)


def custom_processing_function(
        frame, 
        tracking_id, 
        obj_coord, 
        **proc_kwargs
    ) -> tuple[tuple, tuple]:
    xmin, ymin, xmax, ymax = obj_coord

    # If number of processed objects is more than max limit,
    # delete the first to enter.
    if len(proc_kwargs['proc_obj_info'])>proc_kwargs['max_n_obj']:
        if proc_kwargs['last_obj_to_del'] not in proc_kwargs['proc_obj_info']:
            proc_kwargs['last_obj_to_del'] += 1
        else:
            del proc_kwargs['proc_obj_info'][proc_kwargs['last_obj_to_del']]

    # If there are no processed objects set the first to enter id
    # as last_obj_to_del.
    if len(proc_kwargs['proc_obj_info']) == 0:
        proc_kwargs['last_obj_to_del'] = tracking_id

    # If new tracking id, create new entry in the processed objects dict.
    if tracking_id>proc_kwargs['prev_id']:
        proc_kwargs['prev_id'] = tracking_id
        proc_kwargs['proc_obj_info'][tracking_id] = [1, [], [], False]
        # Processing object information dict list element meaning:
        # 0 - counter,
        # 1 - custom intermediate data accumulated until final processing,
        # 2 - data assigned after final processing,
        # 3 - if object has 2 states, e.g. recognized, unrecognized

    # If the number of performed processings on a object with a certain
    # tracking id is less than needed, do another processing.
    if proc_kwargs['proc_obj_info'][tracking_id][0]<=proc_kwargs['n_det']:

        # DO SOMETHING

        proc_kwargs['proc_obj_info'][tracking_id][1].append(plate_number)

        idx = int(proc_kwargs['proc_obj_info'][tracking_id][0] \
            * (len(proc_kwargs['proc_animation']) / proc_kwargs['n_det']))

        proc_kwargs['proc_obj_info'][tracking_id][2] \
            = proc_kwargs['proc_animation'][idx - 1 \
                if idx==len(proc_kwargs['proc_animation']) else idx]

        proc_kwargs['proc_obj_info'][tracking_id][0] += 1

    # If the number of performed processings on a object with a certain
    # tracking id has sufficed do final processing.
    if proc_kwargs['proc_obj_info'][tracking_id][0]==proc_kwargs['n_det']+1:

        # DO SOMETHING

        proc_kwargs['proc_obj_info'][tracking_id][0] += 1

    output = (
        (
            proc_kwargs['proc_obj_info'], 
            proc_kwargs['last_obj_to_del'], 
            proc_kwargs['prev_id']
        ),
        (
            False, 
            proc_kwargs['proc_obj_info'][tracking_id][3],
            ((0,255,0),(0,0,255)),
            proc_kwargs['proc_obj_info'][tracking_id][2]
        )
    )

    return output


prev_id = -1
n_det = 10
max_n_obj = 50
last_obj_to_del = 0
proc_obj_info = {}
proc_animation = {i:'|'*(i+1) for i in range(8)}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        break

    output = mot_tracker.track_objects(
        frame,
        custom_processing_function,
        prev_id=prev_id,
        n_det=n_det,
        max_n_obj=max_n_obj,
        last_obj_to_del=last_obj_to_del,
        proc_animation=proc_animation,
        # Custom kwargs
    )

    tracked_ojb_info, frame_with_bboxes, proc_obj_info_tuple = output

    if None not in proc_obj_info_tuple:
        proc_obj_info, last_obj_to_del, prev_id = proc_obj_info_tuple

    cv2.imshow('Object tracking with deepSORT', frame_with_bboxes)
```

## [Changelog](https://github.com/simsiru/pymot/blob/main/CHANGELOG.md)

## License

[MIT](https://choosealicense.com/licenses/mit/)
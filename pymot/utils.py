import cv2
import numpy as np


def draw_bbox_tracking(
    coords: tuple,
    height: int,
    frame: np.ndarray,
    tracking_id: int,
    class_name: str = "",
    rand_colors: bool = True,
    rec_bool: bool = False,
    colors: tuple = ((0, 255, 0), (0, 0, 255)),
    info_text: str = "",
    unknown_obj_info: str = "UNKNOWN",
) -> None:
    """Function for visualizing tracking bounding boxes"""

    assert len(colors) == 2, "Colors must a list of tuples of length 2"

    assert len(coords) == 4, "Bounding box coordinates must be of 4 values"

    xmin, ymin, xmax, ymax = coords

    if rand_colors:
        np.random.seed(tracking_id)
        r = np.random.rand()
        g = np.random.rand()
        b = np.random.rand()
        color = (int(r * 255), int(g * 255), int(b * 255))
    else:
        if rec_bool:
            color = colors[0]
        else:
            info_text = unknown_obj_info
            color = colors[1]

    cv2.rectangle(frame, (xmin, ymax), (xmax, ymin), color, 2)

    text = "{} ID: {} [{}]".format(class_name, tracking_id, info_text)

    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_DUPLEX, 0.75, thickness=1
    )

    if (ymax + text_height) > height:
        cv2.rectangle(
            frame,
            (xmin, ymax),
            (xmin + text_width, ymax - text_height - baseline),
            color,
            thickness=cv2.FILLED,
        )

        cv2.putText(
            frame,
            text,
            (xmin, ymax - 4),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )
    else:
        cv2.rectangle(
            frame,
            (xmin, ymax + text_height + baseline),
            (xmin + text_width, ymax),
            color,
            thickness=cv2.FILLED,
        )

        cv2.putText(
            frame,
            text,
            (xmin, ymax + text_height + 3),
            cv2.FONT_HERSHEY_DUPLEX,
            0.75,
            (0, 0, 0),
            1,
            lineType=cv2.LINE_AA,
        )

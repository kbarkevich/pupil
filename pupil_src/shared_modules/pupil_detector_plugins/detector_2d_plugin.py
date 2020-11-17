"""
(*)~---------------------------------------------------------------------------
Pupil - eye tracking platform
Copyright (C) 2012-2020 Pupil Labs

Distributed under the terms of the GNU
Lesser General Public License (LGPL v3.0).
See COPYING and COPYING.LESSER for license details.
---------------------------------------------------------------------------~(*)
"""
import logging

from pupil_detectors import Detector2D, DetectorBase, Roi
from pyglui import ui
from pyglui.cygl.utils import draw_gl_texture

import numpy as np

import glfw
from gl_utils import (
    adjust_gl_view,
    basic_gl_setup,
    clear_gl_screen,
    make_coord_system_norm_based,
    make_coord_system_pixel_based,
)
from methods import normalize
from plugin import Plugin

from .detector_base_plugin import PropertyProxy, PupilDetectorPlugin
from .visualizer_2d import draw_pupil_outline

from RITnet.image import get_mask_from_PIL_image, init_model, get_pupil_ellipse_from_PIL_image

logger = logging.getLogger(__name__)


class Detector2DPlugin(PupilDetectorPlugin):
    uniqueness = "by_class"
    icon_font = "pupil_icons"
    icon_chr = chr(0xEC18)

    label = "C++ 2d detector"
    identifier = "2d"
    order = 0.100

    def __init__(
        self, g_pool=None, namespaced_properties=None, detector_2d: Detector2D = None
    ):
        super().__init__(g_pool=g_pool)
        self.detector_2d = detector_2d or Detector2D(namespaced_properties or {})
        self.proxy = PropertyProxy(self.detector_2d)
        self.model = init_model(modelname="best_model.pkl")
        self.model_channels = 4
        
    def detect(self, frame, **kwargs):
        # convert roi-plugin to detector roi
        roi = Roi(*self.g_pool.roi.bounds)
        useRITnet = self.g_pool.ritnet_2d
        
        img = frame.gray
        if useRITnet:
            ellipsedata = get_pupil_ellipse_from_PIL_image(img, self.model, trim_pupil=False, channels=self.model_channels)
            # img = np.uint8(get_mask_from_PIL_image(img, self.model) * 255)
            if ellipsedata is not None:
                result = {}
                ellipse = {}
                ellipse["center"] = (ellipsedata[0], ellipsedata[1])
                ellipse["axes"] = (ellipsedata[2]*2, ellipsedata[3]*2)
                ellipse["angle"] = ellipsedata[4]
                result["ellipse"] = ellipse
                result["diameter"] = ellipsedata[2]*2
                result["location"] = ellipse["center"]
                result["confidence"] = 0.99
            else:
                result = {}
                ellipse = {}
                ellipse["center"] = (0.0, 0.0)
                ellipse["axes"] = (0.0, 0.0)
                ellipse["angle"] = 0.0
                result["ellipse"] = ellipse
                result["diameter"] = 0.0
                result["location"] = ellipse["center"]
                result["confidence"] = 0.0
        else:
            debug_img = frame.bgr if self.g_pool.display_mode == "algorithm" else None
            result = self.detector_2d.detect(
                gray_img=img,
                color_img=debug_img,
                roi=roi,
            )
        
        # print("-----------------")
        # for key, value in result.items():
        #     #print(key + ": " + str(type(value)))
        #     if not isinstance(value, dict) and not isinstance(value, bytes):
        #         print(key + ": " + str(value))
        #     elif isinstance(value, dict):
        #         print(key + ":")
        #         for key2, value2 in value.items():
        #             print("- " + key2 + ": " + str(value2))
        #     else:
        #         print(key + ": " + str(type(value)))
        eye_id = self.g_pool.eye_id
        location = result["location"]
        result["norm_pos"] = normalize(
            location, (frame.width, frame.height), flip_y=True
        )
        result["timestamp"] = frame.timestamp
        result["topic"] = f"pupil.{eye_id}.{self.identifier}"
        result["id"] = eye_id
        result["method"] = "2d c++"
        #result["previous_detection_results"] = result.copy()
        return result

    @property
    def pupil_detector(self) -> DetectorBase:
        return self.detector_2d

    @property
    def pretty_class_name(self):
        return "Pupil Detector 2D"

    def gl_display(self):
        if self._recent_detection_result:
            draw_pupil_outline(self._recent_detection_result, color_rgb=(0, 0.5, 1))

    def init_ui(self):
        super().init_ui()
        self.menu.label = self.pretty_class_name
        self.menu_icon.label_font = "pupil_icons"
        info = ui.Info_Text(
            "Switch to the algorithm display mode to see a visualization of pupil detection parameters overlaid on the eye video. "
            + "Adjust the pupil intensity range so that the pupil is fully overlaid with blue. "
            + "Adjust the pupil min and pupil max ranges (red circles) so that the detected pupil size (green circle) is within the bounds."
        )
        self.menu.append(info)
        self.menu.append(
            ui.Slider(
                "2d.intensity_range",
                self.proxy,
                label="Pupil intensity range",
                min=0,
                max=60,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "2d.pupil_size_min",
                self.proxy,
                label="Pupil min",
                min=1,
                max=250,
                step=1,
            )
        )
        self.menu.append(
            ui.Slider(
                "2d.pupil_size_max",
                self.proxy,
                label="Pupil max",
                min=50,
                max=400,
                step=1,
            )
        )
        self.menu.append(
            ui.Switch(
                "ritnet_2d",
                self.g_pool,
                label="Enable RITnet"
            )
        )

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt

from ... import __version__
from ...control.config_manager import config
from ...definitions import (
    Color3f,
    LabelingMode,
    ObjectDetectionFormat,
    SemanticSegmentationFormat,
)
from ...definitions.label_formats.base import BaseLabelFormat
from ...utils.color import hex_to_rgb, rgb_to_hex
from ...utils.logger import warn_once
from ...utils.singleton import SingletonABCMeta
from .exceptions import (
    DefaultIdMismatchException,
    LabelClassNameEmpty,
    LabelIdsNotUniqueException,
    UnknownLabelFormat,
    ZeroLabelException,
)


@dataclass
class ClassConfig:
    name: str
    id: int
    color: Color3f
    top_level_object: Optional[str] = None
    new_top_level_label: Optional[str] = None
    acted_on_object: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "ClassConfig":
        return cls(
            name=data["name"], 
            id=data["id"], 
            color=hex_to_rgb(data["color"]),
            top_level_object=data.get("top_level_object"),
            new_top_level_label=data.get("new_top_level_label"),
            acted_on_object=data.get("acted_on_object")
        )

    def to_dict(self) -> dict:
        result = {
            "name": self.name,
            "id": self.id,
            "color": rgb_to_hex(self.color),
        }
        if self.top_level_object is not None:
            result["top_level_object"] = self.top_level_object
        if self.new_top_level_label is not None:
            result["new_top_level_label"] = self.new_top_level_label
        if self.acted_on_object is not None:
            result["acted_on_object"] = self.acted_on_object
        return result


class LabelConfig(object, metaclass=SingletonABCMeta):
    def __init__(self) -> None:
        self.classes: List[ClassConfig]
        self.default: int
        self.type: LabelingMode
        self.format: BaseLabelFormat
        self._current_config_path: Optional[Path] = None  # Track which config file is currently loaded

        if getattr(self, "_loaded", False) != True:
            self.load_config()

    def load_config(self, class_definition_path: Optional[Path] = None) -> None:
        if class_definition_path is None:
            class_definition_path = config.getpath("FILE", "class_definitions")
            
        self._current_config_path = class_definition_path
        
        if class_definition_path.exists():
            with class_definition_path.open("r") as stream:
                data = json.load(stream)

            self.classes = [ClassConfig.from_dict(c) for c in data["classes"]]
            self.default = data["default"]
            self.type = LabelingMode(data["type"])
            self.format = data["format"]
        else:
            self.classes = [ClassConfig("cart", 0, color=Color3f(1, 0, 0))]
            self.default = 0
            self.type = LabelingMode.OBJECT_DETECTION
            self.format = ObjectDetectionFormat.CENTROID_REL
        self.validate()
        self._loaded = True

    def load_config_for_pointcloud(self, pcd_path: Path) -> bool:
        """
        Load class definitions for a specific point cloud file.
        Returns True if a specific config was found and loaded, False if fallback was used.
        """
        # Get the label folder from config
        label_folder = config.getpath("FILE", "label_folder")
        
        # Construct the path for the PLY-specific class definitions
        ply_specific_config = label_folder / f"{pcd_path.stem.replace('laser_scan', 'classes')}.json"
        
        if ply_specific_config.exists():
            # Load PLY-specific class definitions
            self.load_config(ply_specific_config)
            return True
        else:
            # Fallback to default class definitions
            self.load_config()
            return False

    def save_config(self, save_path: Optional[Path] = None) -> None:
        self.validate()
        data = {
            "classes": [c.to_dict() for c in self.classes],
            "default": self.default,
            "type": self.type.value,
            "format": self.format,
            "created_with": {"name": "labelCloud", "version": __version__},
        }
        
        # Use the specified path, or the current config path, or fall back to default
        if save_path is None:
            save_path = self._current_config_path or config.getpath("FILE", "class_definitions")
            
        with save_path.open("w") as stream:
            json.dump(data, stream, indent=4)

    @property
    def nb_of_classes(self) -> int:
        return len(self.classes)

    @property
    def color_map(self) -> npt.NDArray[np.float32]:
        """An (N, 3) array where N is the number of classes and color_map[i] represents the i-th class' rgb color."""
        return np.array([c.color[0:3] for c in self.classes]).astype(np.float32)

    @property
    def class_order(self) -> npt.NDArray[np.int8]:
        """An array lookup table to look up the order of a class id in the label definition."""
        max_class_id = max(c.id for c in self.classes) + 1
        lookup = -np.ones((max_class_id,), dtype=np.int8)
        for order, c in enumerate(self.classes):
            lookup[c.id] = order
        return lookup

    # GETTERS

    def get_classes(self) -> Dict[str, ClassConfig]:
        return {c.name: c for c in self.classes}

    def get_class(self, class_name: str) -> ClassConfig:
        return self.get_classes()[class_name]

    def get_relative_class(self, current_class: str, step: int) -> str:
        """Get class, relative to current by id according to given step"""
        if step == 0:
            return current_class
        id2name = {cc.id: cc.name for cc in self.classes}
        name2id = {v: k for k, v in id2name.items()}
        ids = name2id.values()
        corner_case_id = max(ids) if step < 0 else min(ids)
        current_id = name2id[current_class]
        result_id = current_id + step
        result_id = result_id if result_id in ids else corner_case_id
        return id2name[result_id]

    def get_class_color(self, class_name: str) -> Color3f:
        try:
            return self.get_classes()[class_name].color
        except KeyError:
            warn_once(
                "No color defined for class '%s'!" "Proceeding with red.", class_name
            )
            return hex_to_rgb("#FF0000")

    def has_valid_default_class(self) -> bool:
        for c in self.classes:
            if c.id == self.default:
                return True
        return False

    def get_default_class_name(self) -> str:
        for c in self.classes:
            if c.id == self.default:
                return c.name
        raise DefaultIdMismatchException(
            f"Default class id `{self.default}` is missing in the class list."
        )

    # SETTERS

    def set_first_as_default(self) -> None:
        self.default = self.classes[0].id

    def set_default_class(self, class_name: str) -> None:
        self.default = next((c.id for c in self.classes if c.name == class_name))
        self.save_config()

    def set_class_color(self, class_name: str, color: Color3f) -> None:
        self.get_class(class_name).color = color
        self.save_config()

    def set_label_format(self, label_format: Union[BaseLabelFormat, str]) -> None:
        if label_format not in {
            *ObjectDetectionFormat.list(),
            *SemanticSegmentationFormat.list(),
        }:
            raise UnknownLabelFormat(label_format)

        self.format = label_format  # type: ignore

    # VALIDATION
    def validate(self) -> None:
        if self.nb_of_classes == 0:
            raise ZeroLabelException("At least one label required.")
        # validate the default id presents in the classes
        self.get_default_class_name()
        # validate the ids are unique
        if len({c.id for c in self.classes}) != self.nb_of_classes:
            raise LabelIdsNotUniqueException("Class ids are not unique.")

        for label_class in self.classes:
            if label_class.name == "":
                raise LabelClassNameEmpty("At least one class name is empty.")

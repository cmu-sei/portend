#
# Portend Toolset
#
# Copyright 2024 Carnegie Mellon University.
#
# NO WARRANTY. THIS CARNEGIE MELLON UNIVERSITY AND SOFTWARE ENGINEERING INSTITUTE MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. CARNEGIE MELLON UNIVERSITY MAKES NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED, AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF THE MATERIAL. CARNEGIE MELLON UNIVERSITY DOES NOT MAKE ANY WARRANTY OF ANY KIND WITH RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.
#
# Licensed under a MIT (SEI)-style license, please see license.txt or contact permission@sei.cmu.edu for full terms.
#
# [DISTRIBUTION STATEMENT A] This material has been approved for public release and unlimited distribution.  Please see Copyright notice for non-US Government use and distribution.
#
# This Software includes and/or makes use of Third-Party Software each subject to its own license.
#
# DM24-1299
#

from __future__ import annotations

import json
import os
import typing
from pathlib import Path
from typing import Any, Optional, Type

import cv2
import numpy.typing as npt
from natsort import natsorted

from portend.datasets import dataset
from portend.datasets.dataset_loader import load_dataset_class

DEFAULT_EXTENSIONS = [".png", ".jpg", ".tif", ".tiff"]
DEFAULT_IMAGE_DATASET_CLASS = "portend.datasets.image_dataset.ImageDataSet"


def load_image_dataset_class(
    dataset_class_name: Optional[str],
) -> Type[ImageDataSet]:
    """Loads theg given image dataset class, or the default one."""
    if dataset_class_name is None:
        dataset_class_name = DEFAULT_IMAGE_DATASET_CLASS
    return typing.cast(
        Type[ImageDataSet], load_dataset_class(dataset_class_name)
    )


def create_image_dataset_to_file(
    dataset_class_name: Optional[str],
    image_folder: Optional[str],
    extensions: Optional[list[str]] = DEFAULT_EXTENSIONS,
    fields: Optional[dict[str, Any]] = {},
    existing_data: Optional[str] = "",
    json_file: Optional[str] = None,
):
    """Creates an image dataset from a config and stores it to a file."""
    if image_folder is None:
        raise Exception("Source folder for images was not provided.")

    # First load the type of image dataset.
    dataset_class = load_image_dataset_class(dataset_class_name)

    # Now load it from a set of images.
    dataset: ImageDataSet = dataset_class.create_dataset_from_folder(
        image_folder,
        extensions,
        fields,
        existing_data,
    )

    # Finally, store dataset info to JSON file.
    output_filepath = (
        json_file
        if json_file is not None
        else os.path.join(image_folder, ImageDataSet.DEFAULT_DATASET_FILENAME)
    )
    dataset.save_to_file(output_filepath, save_images=False)


class ImageDataSet(dataset.DataSet):
    """A dataset for handling a list of images."""

    DEFAULT_IMAGE_FOLDER = "."
    DEFAULT_IMAGE_PATH_KEY = "image_path"
    DEFAULT_DATASET_FILENAME = "dataset.json"

    def __init__(self, image_folder: str = DEFAULT_IMAGE_FOLDER):
        """Inits, receives output folder for images."""
        self.image_path_key: str = ImageDataSet.DEFAULT_IMAGE_PATH_KEY
        self.image_folder = image_folder
        self.image_list: list[npt.NDArray[Any]] = []
        self.image_names: list[str] = []

    def set_image_folder(self, image_folder: Optional[str]):
        self.image_folder = (
            image_folder
            if image_folder is not None
            else ImageDataSet.DEFAULT_IMAGE_FOLDER
        )

    # Overriden
    def clone_image_dataset(self, cloned_dataset: ImageDataSet) -> ImageDataSet:
        """Clones into the provided dataset."""
        cloned_dataset = typing.cast(
            ImageDataSet, super().clone(cloned_dataset)
        )
        cloned_dataset.image_path_key = self.image_path_key
        cloned_dataset.image_folder = self.image_folder
        cloned_dataset.image_list = (
            self.image_list
        )  # TODO: deep copy of this. Or does it waste too much space?
        cloned_dataset.image_names = self.image_names.copy()
        return cloned_dataset

    # Overriden.
    def post_process(self, dataset_config: dict[str, Any]):
        """Loads the actual images from the filenames or folders indicated in the JSON file."""
        # Goes over the paths and loads the actual images into a list.
        self.image_path_key = dataset_config.get(
            "dataset_image_path_key", ImageDataSet.DEFAULT_IMAGE_PATH_KEY
        )
        self.image_list = [
            cv2.imread(img_file)
            for img_file in self.dataframe[self.image_path_key]
        ]
        self.image_names = [
            str(Path(img_file).stem)
            for img_file in self.dataframe[self.image_path_key]
        ]

        # Assuming all images are in the same folder, set the image folder to the one of the first image (this is mostly informational only).
        if len(self.dataframe) > 0:
            self.set_image_folder(
                str(Path(self.dataframe[self.image_path_key][0]).parent)
            )

    def set_images_from_list(
        self, images: list[npt.NDArray[Any]], image_names: list[str] = []
    ):
        """Adds the given images to the internal dataset structure. Assumes there are samples for each image already."""
        if self.get_number_of_samples() != len(images) or len(images) != len(
            image_names
        ):
            raise RuntimeError(
                f"Can't set images from list, lengths don't match: samples: {self.get_number_of_samples()}, images: {len(images)}, image names: {len(image_names)}"
            )

        self.image_list = images
        self.image_names = image_names

        # Replace value of image path with the new ones from the list.
        for i, _ in self.dataframe.iterrows():
            i = typing.cast(int, i)
            self.dataframe.at[i, self.image_path_key] = os.path.join(
                self.image_folder, self.image_names[i]
            )

    # Overriden.
    def save_to_file(self, output_filename: str, save_images: bool = True):
        """Saves dataset to JSON file, and images to configured folder."""
        # Call base to save the paths fo a JSON file.
        super().save_to_file(output_filename)

        if save_images:
            # Write the images to disk.
            os.makedirs(self.image_folder, exist_ok=True)
            for i in range(len(self.image_list)):
                cv2.imwrite(
                    self.dataframe.loc[i][self.image_path_key],
                    self.image_list[i],
                )

    @classmethod
    def create_dataset_from_folder(
        cls,
        image_folder: Optional[str],
        extensions: Optional[list[str]] = DEFAULT_EXTENSIONS,
        fields: Optional[dict[str, Any]] = {},
        existing_data: Optional[str] = "",
    ) -> ImageDataSet:
        """Creates a JSON file with the expected ImageDataSet format for all images in a given folder."""
        if image_folder is None:
            raise Exception("Did not receive image folder value")
        if extensions is None:
            extensions = DEFAULT_EXTENSIONS

        # Get the files in the folder and structure their paths in a dict appropriate for the dataset.
        print(f"Creating ImageDataset dataset from folder {image_folder}")
        extensions_lower = [x.lower() for x in extensions]
        files = natsorted(os.listdir(image_folder))
        file_paths = [os.path.join(image_folder, file) for file in files]

        samples = [
            {
                dataset.DataSet.DEFAULT_ID_KEY: os.path.basename(
                    image_file_path
                ),
                ImageDataSet.DEFAULT_IMAGE_PATH_KEY: image_file_path,
            }
            for image_file_path in file_paths
            if len(extensions) == 0
            or Path(image_file_path).suffix.lower() in extensions_lower
        ]

        # Add additional fields, if any.
        if fields is not None and len(fields) > 0:
            for sample in samples:
                for field_key, field_value in fields.items():
                    sample.update({field_key: field_value})

        # Merge with existing data if any.
        if existing_data is not None and len(existing_data) > 0:
            with open(existing_data) as f:
                data = json.load(f)
                for sample in samples:
                    for item in data:
                        if (
                            sample[dataset.DataSet.DEFAULT_ID_KEY]
                            == item[dataset.DataSet.DEFAULT_ID_KEY]
                        ):
                            curr_data = item[dataset.DataSet.DEFAULT_ID_KEY]
                            print(curr_data)
                            for field_key, field_value in curr_data.items():
                                sample.update({field_key: field_value})

        # Create an image dataset with this.
        image_dataset = ImageDataSet()
        image_dataset.set_samples(samples)
        return image_dataset

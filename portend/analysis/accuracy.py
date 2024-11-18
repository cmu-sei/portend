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

from typing import Any, Optional

import numpy.typing as npt
from sklearn.metrics import confusion_matrix


class ClassificationAccuracy:
    """Class used to calculate false positives and negatives, and accuracy for the given samples."""

    ACCURACY_TRUE_POSITIVE = "tp"
    ACCURACY_TRUE_NEGATIVE = "tn"
    ACCURACY_FALSE_POSITIVE = "fp"
    ACCURACY_FALSE_NEGATIVE = "fn"

    tf_pn_by_sample: Optional[list[str]] = None
    total_true_positives = 0
    total_true_negatives = 0
    total_false_positives = 0
    total_false_negatives = 0

    def _calculate_true_false_positives_negatives(
        self, expected_results, predictions, labels, positive_class
    ):
        """Calculates confusion matrix, and for each sample if it was a true/false positive/negative."""
        if expected_results is not None and predictions is not None:
            conf_matrix: npt.NDArray[Any] = confusion_matrix(
                expected_results, predictions, labels=labels
            )
            (
                self.total_true_negatives,
                self.total_false_positives,
                self.total_false_negatives,
                self.total_true_positives,
            ) = conf_matrix.ravel()
            # print(f"TN: {self.total_true_negatives}, TP: {self.total_true_positives}, "
            #      f"FN: {self.total_false_negatives}, FP: {self.total_false_positives}")

            self.tf_pn_by_sample = []
            for idx, truth in enumerate(expected_results):
                if truth == predictions[idx]:
                    if truth == positive_class:
                        self.tf_pn_by_sample.append(self.ACCURACY_TRUE_POSITIVE)
                    else:
                        self.tf_pn_by_sample.append(self.ACCURACY_TRUE_NEGATIVE)
                else:
                    if truth == positive_class:
                        self.tf_pn_by_sample.append(
                            self.ACCURACY_FALSE_POSITIVE
                        )
                    else:
                        self.tf_pn_by_sample.append(
                            self.ACCURACY_FALSE_NEGATIVE
                        )

    def calculate_accuracy(
        self, expected_results, predictions, labels, positive_class
    ) -> float:
        """Calculates the accuracy and returns it"""
        if self.tf_pn_by_sample is None:
            self._calculate_true_false_positives_negatives(
                expected_results, predictions, labels, positive_class
            )

        return (self.total_true_negatives + self.total_true_positives) / (
            self.total_true_negatives
            + self.total_true_positives
            + self.total_false_positives
            + self.total_false_negatives
        )

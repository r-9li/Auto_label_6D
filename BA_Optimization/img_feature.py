from collections import defaultdict
from numbers import Number

import numpy as np
from params import BA_param


class ImageFeature(object):
    def __init__(self, image):
        self.image = image
        self.height, self.width = image.shape[:2]

        self.keypoints = []  # list of cv2.KeyPoint
        self.descriptors = []  # numpy.ndarray

        self.extractor = BA_param.extractor
        self.matcher = BA_param.descriptor_matcher

        self.matching_distance = BA_param.matching_distance
        self.neighborhood = (
                BA_param.matching_cell_size * BA_param.matching_neighborhood)

    def extract(self):
        self.keypoints, self.descriptors = self.extractor.detectAndCompute(self.image, None)

        self.unmatched = np.ones(len(self.keypoints), dtype=bool)

    def find_matches(self, predictions, descriptors):
        matches = dict()
        distances = defaultdict(lambda: float('inf'))
        for m, query_idx, train_idx in self.matched_by(descriptors):
            if m.distance > min(distances[train_idx], self.matching_distance):
                continue

            pt1 = predictions[query_idx]
            pt2 = self.keypoints[train_idx].pt
            dx = pt1[0] - pt2[0]
            dy = pt1[1] - pt2[1]
            if np.sqrt(dx * dx + dy * dy) > self.neighborhood:
                continue

            matches[train_idx] = query_idx
            distances[train_idx] = m.distance
        matches = [(i, j) for j, i in matches.items()]
        return matches

    def matched_by(self, descriptors):
        unmatched_descriptors = self.descriptors[self.unmatched]
        if len(unmatched_descriptors) == 0:
            return []
        lookup = dict(zip(
            range(len(unmatched_descriptors)),
            np.where(self.unmatched)[0]))

        matches = self.matcher.match(
            np.array(descriptors), unmatched_descriptors)
        return [(m, m.queryIdx, m.trainIdx) for m in matches]

    def get_keypoint(self, i):
        return self.keypoints[i]

    def get_descriptor(self, i):
        return self.descriptors[i]

    def get_color(self, pt):
        x = int(np.clip(pt[0], 0, self.width - 1))
        y = int(np.clip(pt[1], 0, self.height - 1))
        color = self.image[y, x]
        if isinstance(color, Number):
            color = np.array([color, color, color])
        return color[::-1] / 255.

    def set_matched(self, i):

        self.unmatched[i] = False

    def get_unmatched_keypoints(self):
        keypoints = []
        descriptors = []
        indices = []

        for i in np.where(self.unmatched)[0]:
            keypoints.append(self.keypoints[i])
            descriptors.append(self.descriptors[i])
            indices.append(i)

        return keypoints, descriptors, indices

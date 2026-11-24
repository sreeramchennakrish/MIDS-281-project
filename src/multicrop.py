import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.preprocessing import StandardScaler, normalize

CROP_SIZE = 128


def load_rgb(path):
    return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)


def make_crops(img, size=CROP_SIZE):
    h, w = img.shape[:2]
    m = min(h, w)
    crops = []
    for t in (0.0, 0.5, 1.0):
        y, x = round((h - m) * t), round((w - m) * t)
        crops.append(cv2.resize(img[y:y + m, x:x + m], (size, size), interpolation=cv2.INTER_AREA))
    return crops


def stretch(img, size=CROP_SIZE):
    return [cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)]


def histogram(img, ranges, bins=16):
    parts = []
    for ch, rng in enumerate(ranges):
        h = cv2.calcHist([img], [ch], None, [bins], list(rng)).ravel()
        parts.append(h / (h.sum() + 1e-8))
    return np.concatenate(parts)


def crop_features(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    rgb = histogram(crop, [(0, 256)] * 3)
    hsv = histogram(cv2.cvtColor(crop, cv2.COLOR_RGB2HSV), [(0, 180), (0, 256), (0, 256)])
    lbp_img = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp, _ = np.histogram(lbp_img, bins=np.arange(11))
    lbp = lbp / (lbp.sum() + 1e-8)
    hog_vec = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm="L2-Hys")
    return rgb, hsv, lbp, hog_vec


def build_features(paths, views=make_crops):
    """Returns the 4 feature groups (one row per view) and the image index of each row."""
    groups = [[], [], [], []]
    image_ids = []
    for i, path in enumerate(paths):
        for crop in views(load_rgb(path)):
            for g, f in zip(groups, crop_features(crop)):
                g.append(f)
            image_ids.append(i)
    return [np.array(g, dtype=np.float32) for g in groups], np.array(image_ids)


class GroupScaler:
    """Same as notebook 02: standardize each group on train, then L2-normalize each row within the group."""

    def fit(self, groups):
        self.scalers = [StandardScaler().fit(g) for g in groups]
        return self

    def transform(self, groups):
        return np.hstack([normalize(s.transform(g)) for s, g in zip(self.scalers, groups)])


def predict_vote(model, X, image_ids):
    """Sum each image's crop scores, then pick the highest class."""
    scores = model.decision_function(X)
    totals = np.zeros((image_ids.max() + 1, scores.shape[1]))
    np.add.at(totals, image_ids, scores)
    return totals.argmax(axis=1)

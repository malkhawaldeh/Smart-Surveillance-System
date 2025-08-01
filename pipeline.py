import os
import json
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, resnet34, ResNet50_Weights, ResNet34_Weights
import torch.nn.functional as F

# ================= CONFIGURATION & PATHS =================

# --- Attribute model weights ---
CELEBA_WEIGHTS = "weights/best_celeba_model.pth"
FAIRFACE_WEIGHTS = "weights/best_fairface_model.pth"
FASHIONPEDIA_WEIGHTS = "weights/best_fashionpedia_model.pth"
UPAR_WEIGHTS = "weights/best_upar_model.pth"

# --- Override/config JSONs ---
CELEBA_ATTRS_JSON = "configs/celeba_attrs_list.json"
CELEBA_THRESHOLDS_JSON = "configs/celeba_thresholds.json"
FAIRFACE_LABELS_JSON = "configs/fairface_labels.json"
FASHIONPEDIA_ATTRS_JSON = "configs/fashionpedia_attrs.json"
FASHIONPEDIA_THRESHOLD_JSON = "configs/fashionpedia_threshold.json"
UPAR_LABELS_JSON = "configs/upar_labels.json"

# --- Detection model files ---
FACE_PROTO = "models/deploy.prototxt"
FACE_MODEL = "models/res10_300x300_ssd_iter_140000.caffemodel"
PERSON_PROTO = "models/MobileNetSSD_deploy.prototxt"
PERSON_MODEL = "models/mobilenet_iter_73000.caffemodel"

# --- Threshold defaults ---
FACE_DETECTION_CONF = 0.4
PERSON_DETECTION_CONF = 0.5
FASHIONPEDIA_PRED_CONF_DEFAULT = 0.5

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================== FALLBACK LABELS & THRESHOLDS ================
CELEBA_ATTRIBUTES_CLEANED = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Bags_Under_Eyes", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair",
    "Blond_Hair", "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin", "Eyeglasses", "Goatee",
    "Heavy_Makeup", "High_Cheekbones", "Male", "Mouth_Slightly_Open", "Narrow_Eyes", "No_Beard", "Oval_Face",
    "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks", "Sideburns", "Smiling", "Straight_Hair",
    "Wavy_Hair", "Wearing_Earrings", "Wearing_Lipstick", "Wearing_Necklace", "Wearing_Necktie", "Young"
]
CELEBA_THRESHOLDS_DEFAULT = {
    "Male": 0.95,
    "Smiling": 0.50,
    "Wearing_Earrings": 0.50,
    "Heavy_Makeup": 0.75,
    "No_Beard": 0.95,
    "Eyeglasses": 0.70,
    "Young": 0.70,
}

FAIRFACE_RACE_LABELS = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']
FAIRFACE_GENDER_LABELS = ['Female', 'Male']
FAIRFACE_AGE_LABELS = ['0-2', '10-19', '20-29', '3-9', '30-39', '40-49', '50-59', '60-69', 'more than 70']
FAIRFACE_THRESHOLDS_DEFAULT = {"race": 0.50, "gender": 0.2, "age": 0.2}

UPAR_FULL_LABEL_LIST = [
    'Accessory-Backpack', 'Accessory-Bag', 'Accessory-Glasses-Normal', 'Accessory-Hat', 'Age-Adult',
    'Age-Young', 'Gender-Female', 'Hair-Length-Long', 'Hair-Length-Short', 'LowerBody-Color-Black',
    'LowerBody-Color-Blue', 'LowerBody-Color-Brown', 'LowerBody-Color-Grey', 'LowerBody-Color-Other',
    'LowerBody-Color-White', 'LowerBody-Length-Short', 'LowerBody-Type-Skirt&Dress',
    'LowerBody-Type-Trousers&Shorts', 'UpperBody-Color-Black', 'UpperBody-Color-Blue', 'UpperBody-Color-Brown',
    'UpperBody-Color-Green', 'UpperBody-Color-Grey', 'UpperBody-Color-Other', 'UpperBody-Color-Pink',
    'UpperBody-Color-Purple', 'UpperBody-Color-Red', 'UpperBody-Color-White', 'UpperBody-Color-Yellow',
    'UpperBody-Length-Short'
]
UPAR_LABEL_COLS = [
    'Accessory-Backpack', 'Accessory-Bag', 'Accessory-Glasses-Normal', 'Accessory-Hat',
    'Age-Adult', 'Age-Young', 'Gender-Female', 'Hair-Length-Long', 'Hair-Length-Short'
]
UPAR_THRESHOLDS_DEFAULT = {
    'Accessory-Backpack': 0.80, 'Accessory-Bag': 0.75, 'Accessory-Glasses-Normal': 0.85,
    'Accessory-Hat': 0.85, 'Age-Adult': 0.85, 'Age-Young': 0.85, 'Gender-Female': 0.85,
    'Hair-Length-Long': 0.85, 'Hair-Length-Short': 0.85
}

FASHIONPEDIA_ATTRIBUTES_FALLBACK = {}  # id -> name

# ================== TRANSFORMS =====================
celeba_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
fairface_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
upar_transform = transforms.Compose([
    transforms.Resize((231, 93)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])
fashionpedia_transform = transforms.Compose([
    transforms.Resize((231, 93)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# ================== MODEL DEFINITIONS =====================
class CelebAModel(nn.Module):
    def __init__(self, num_attrs):
        super().__init__()
        self.model = resnet50(weights=None)
        in_feat = self.model.fc.in_features
        self.model.fc = nn.Linear(in_feat, num_attrs)

    def forward(self, x):
        return self.model(x)


class FairFaceMultiTask(nn.Module):
    def __init__(self):
        super(FairFaceMultiTask, self).__init__()
        try:
            base = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        except Exception:
            from torchvision.models import resnet34
            base = resnet34(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(512, 18)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        out = self.fc(x)
        race_logits = out[:, :7]
        gender_logits = out[:, 7:9]
        age_logits = out[:, 9:]
        return race_logits, gender_logits, age_logits


class FashionpediaModel(nn.Module):
    def __init__(self, num_attributes):
        super(FashionpediaModel, self).__init__()
        try:
            base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            base = resnet50(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_attributes)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


class UPARModel(nn.Module):
    def __init__(self, num_classes=30):
        super(UPARModel, self).__init__()
        try:
            base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        except Exception:
            base = resnet50(weights=None)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
        self.model = base

    def forward(self, x):
        return self.model(x)


# ================== DETECTOR LOADERS =====================
def load_face_detector():
    if not os.path.exists(FACE_PROTO) or not os.path.exists(FACE_MODEL):
        print(f"[WARN] Face detection model missing: {FACE_PROTO}, {FACE_MODEL}. Skipping face detection.")
        return None
    try:
        return cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
    except Exception as e:
        print(f"[WARN] Failed to load face detector: {e}")
        return None


def load_person_detector():
    if not os.path.exists(PERSON_PROTO) or not os.path.exists(PERSON_MODEL):
        print(f"[WARN] Person detection model missing: {PERSON_PROTO}, {PERSON_MODEL}. Skipping person detection.")
        return None
    try:
        return cv2.dnn.readNetFromCaffe(PERSON_PROTO, PERSON_MODEL)
    except Exception as e:
        print(f"[WARN] Failed to load person detector: {e}")
        return None


# ================== JSON / EXPORTED CONFIG LOADERS =====================
def load_json_if_exists(path, default=None):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return default


# ================== MODEL LOADING HELPERS =====================
def load_celeba_model():
    attrs = load_json_if_exists(CELEBA_ATTRS_JSON, CELEBA_ATTRIBUTES_CLEANED)
    thresholds = load_json_if_exists(CELEBA_THRESHOLDS_JSON, CELEBA_THRESHOLDS_DEFAULT)
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(attrs))
    if not os.path.exists(CELEBA_WEIGHTS):
        raise FileNotFoundError(f"CelebA weights not found at {CELEBA_WEIGHTS}")
    state = torch.load(CELEBA_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, attrs, thresholds


def load_fairface_model():
    label_blob = load_json_if_exists(FAIRFACE_LABELS_JSON, None)
    if label_blob:
        race_labels = label_blob.get("race", FAIRFACE_RACE_LABELS)
        gender_labels = label_blob.get("gender", FAIRFACE_GENDER_LABELS)
        age_labels = label_blob.get("age", FAIRFACE_AGE_LABELS)
        thresholds = label_blob.get("thresholds", FAIRFACE_THRESHOLDS_DEFAULT)
    else:
        race_labels = FAIRFACE_RACE_LABELS
        gender_labels = FAIRFACE_GENDER_LABELS
        age_labels = FAIRFACE_AGE_LABELS
        thresholds = FAIRFACE_THRESHOLDS_DEFAULT

    model = FairFaceMultiTask()
    if not os.path.exists(FAIRFACE_WEIGHTS):
        raise FileNotFoundError(f"FairFace weights not found at {FAIRFACE_WEIGHTS}")
    state = torch.load(FAIRFACE_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, race_labels, gender_labels, age_labels, thresholds


def load_fashionpedia_model():
    attrs_blob = load_json_if_exists(FASHIONPEDIA_ATTRS_JSON, None)
    if attrs_blob:
        kept_attrs = {entry["id"]: entry["name"] for entry in attrs_blob}
        ordered_ids = [entry["id"] for entry in attrs_blob]
        if attrs_blob is not None and not ordered_ids:
            print("[WARN] fashionpedia_attrs.json present but empty; falling back to checkpoint introspection")
    else:
        kept_attrs = FASHIONPEDIA_ATTRIBUTES_FALLBACK
        ordered_ids = list(kept_attrs.keys())

    threshold_blob = load_json_if_exists(FASHIONPEDIA_THRESHOLD_JSON, None)
    threshold = threshold_blob.get("threshold", FASHIONPEDIA_PRED_CONF_DEFAULT) if threshold_blob else FASHIONPEDIA_PRED_CONF_DEFAULT

    if not ordered_ids:
        if not os.path.exists(FASHIONPEDIA_WEIGHTS):
            raise FileNotFoundError(f"Fashionpedia weights not found at {FASHIONPEDIA_WEIGHTS}")
        state = torch.load(FASHIONPEDIA_WEIGHTS, map_location=DEVICE)
        out_dim = None
        for key, tensor in state.items():
            if key.endswith("fc.1.weight") or key.endswith("model.fc.1.weight"):
                out_dim = tensor.shape[0]
                break
        if out_dim is None:
            raise RuntimeError("Could not infer Fashionpedia output dimension from checkpoint.")
        ordered_ids = list(range(out_dim))
        kept_attrs = {i: f"attr_{i}" for i in ordered_ids}

    model = FashionpediaModel(num_attributes=len(ordered_ids))
    if not os.path.exists(FASHIONPEDIA_WEIGHTS):
        raise FileNotFoundError(f"Fashionpedia weights not found at {FASHIONPEDIA_WEIGHTS}")
    state = torch.load(FASHIONPEDIA_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, ordered_ids, kept_attrs, threshold

def load_upar_model():
    blob = load_json_if_exists(UPAR_LABELS_JSON, None)
    if blob:
        full_list = blob.get("full_list", UPAR_FULL_LABEL_LIST)
        active_cols = blob.get("active_cols", UPAR_LABEL_COLS)
        thresholds = blob.get("thresholds", UPAR_THRESHOLDS_DEFAULT)
    else:
        full_list = UPAR_FULL_LABEL_LIST
        active_cols = UPAR_LABEL_COLS
        thresholds = UPAR_THRESHOLDS_DEFAULT

    upar_label_idx_map = {label: full_list.index(label) for label in active_cols}
    model = UPARModel(num_classes=len(full_list))
    if not os.path.exists(UPAR_WEIGHTS):
        raise FileNotFoundError(f"UPAR weights not found at {UPAR_WEIGHTS}")
    state = torch.load(UPAR_WEIGHTS, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model, full_list, active_cols, upar_label_idx_map, thresholds


# ================== DETECTION HELPERS =====================
def detect_faces(net, image, conf_threshold=0.5):
    if net is None:
        return []
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        if confidence < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes


def detect_people(net, image, conf_threshold=0.5):
    if net is None:
        return []
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)),
                                 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    boxes = []
    for i in range(detections.shape[2]):
        confidence = float(detections[0, 0, i, 2])
        class_id = int(detections[0, 0, i, 1])
        if class_id != 15 or confidence < conf_threshold:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes


# ================== PREDICTION FUNCTIONS =====================
def predict_celeba(model, pil_face, attrs_list, thresholds):
    face_tensor = celeba_transform(pil_face).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(face_tensor)
        probs = torch.sigmoid(logits).cpu().squeeze()
    pred_attrs = []
    for attr_name, p in zip(attrs_list, probs):
        thresh = thresholds.get(attr_name, 0.8)
        if p.item() > thresh:
            pred_attrs.append(attr_name.replace("_", " "))
    if any(a.lower() == "male" for a in pred_attrs):
        pred_attrs = [a for a in pred_attrs if a.lower() != "male"]
        pred_attrs.insert(0, "Male")
    else:
        pred_attrs.insert(0, "Female")
    return pred_attrs


def predict_fairface(model, pil_face, race_labels, gender_labels, age_labels, thresholds):
    face_tensor = fairface_transform(pil_face).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out_race, out_gender, out_age = model(face_tensor)
        race_probs = F.softmax(out_race, dim=1)[0].cpu()
        gender_probs = F.softmax(out_gender, dim=1)[0].cpu()
        age_probs = F.softmax(out_age, dim=1)[0].cpu()
    texts = []
    if race_probs.max().item() > thresholds["race"]:
        texts.append(race_labels[int(race_probs.argmax().item())])
    if gender_probs.max().item() > thresholds["gender"]:
        texts.append(gender_labels[int(gender_probs.argmax().item())])
    if age_probs.max().item() > thresholds["age"]:
        texts.append(f"Age: {age_labels[int(age_probs.argmax().item())]}")
    return texts


def predict_fashionpedia(model, person_crop, ordered_ids, id_to_name, threshold):
    image_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
    tensor = fashionpedia_transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output).squeeze()
    preds = []
    for idx, attr_id in enumerate(ordered_ids):
        if probs[idx].item() > threshold:
            name = id_to_name.get(attr_id, str(attr_id))
            preds.append(name)
    return preds


def predict_upar(model, person_crop, active_cols, label_idx_map, thresholds):
    image_pil = Image.fromarray(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
    tensor = upar_transform(image_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.sigmoid(output).squeeze()
    preds = []
    for label in active_cols:
        idx = label_idx_map[label]
        if probs[idx].item() > thresholds.get(label, 0.85):
            preds.append(label)
    return preds


# ================== FULL PIPELINE CLASS =====================
class ModelPipeline:
    def __init__(self):
        self.face_net = load_face_detector()
        self.person_net = load_person_detector()

        try:
            self.celeba_model, self.celeba_attrs_list, self.celeba_thresholds = load_celeba_model()
        except Exception as e:
            print(f"[WARNING] CelebA model load failed: {e}")
            self.celeba_model = None
            self.celeba_attrs_list = CELEBA_ATTRIBUTES_CLEANED
            self.celeba_thresholds = CELEBA_THRESHOLDS_DEFAULT

        try:
            (self.fairface_model,
             self.fairface_race_labels,
             self.fairface_gender_labels,
             self.fairface_age_labels,
             self.fairface_thresholds) = load_fairface_model()
        except Exception as e:
            print(f"[WARNING] FairFace model load failed: {e}")
            self.fairface_model = None
            self.fairface_race_labels = FAIRFACE_RACE_LABELS
            self.fairface_gender_labels = FAIRFACE_GENDER_LABELS
            self.fairface_age_labels = FAIRFACE_AGE_LABELS
            self.fairface_thresholds = FAIRFACE_THRESHOLDS_DEFAULT

        try:
            (self.fashionpedia_model,
             self.fashionpedia_ordered_ids,
             self.fashionpedia_id2name,
             self.fashionpedia_threshold) = load_fashionpedia_model()
        except Exception as e:
            print(f"[WARNING] Fashionpedia model load failed: {e}")
            self.fashionpedia_model = None
            self.fashionpedia_ordered_ids = []
            self.fashionpedia_id2name = {}
            self.fashionpedia_threshold = FASHIONPEDIA_PRED_CONF_DEFAULT

        try:
            (self.upar_model,
             self.upar_full_list,
             self.upar_active_cols,
             self.upar_label_idx_map,
             self.upar_thresholds) = load_upar_model()
        except Exception as e:
            print(f"[WARNING] UPAR model load failed: {e}")
            self.upar_model = None
            self.upar_full_list = UPAR_FULL_LABEL_LIST
            self.upar_active_cols = UPAR_LABEL_COLS
            self.upar_label_idx_map = {label: self.upar_full_list.index(label) for label in UPAR_LABEL_COLS}
            self.upar_thresholds = UPAR_THRESHOLDS_DEFAULT

    def infer(self, pil_image: Image.Image):
        cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        h, w = cv_img.shape[:2]
        output = {
            "thresholds": {
                "face_detection_conf": FACE_DETECTION_CONF,
                "person_detection_conf": PERSON_DETECTION_CONF,
                "celeba_thresholds": getattr(self, "celeba_thresholds", CELEBA_THRESHOLDS_DEFAULT),
                "fairface_thresholds": getattr(self, "fairface_thresholds", FAIRFACE_THRESHOLDS_DEFAULT),
                "fashionpedia_threshold": getattr(self, "fashionpedia_threshold", FASHIONPEDIA_PRED_CONF_DEFAULT),
                "upar_thresholds": getattr(self, "upar_thresholds", UPAR_THRESHOLDS_DEFAULT)
            },
            "faces": [],
            "people": []
        }

        # People detection + fashion/UPAR
        person_boxes = []
        if self.person_net is not None:
            person_boxes = detect_people(self.person_net, cv_img, conf_threshold=PERSON_DETECTION_CONF)
        for box in person_boxes:
            x1, y1, x2, y2 = box
            w_box = x2 - x1
            h_box = y2 - y1
            target_ratio = 231 / 93
            padding = int(h_box * 0.05)
            y1_crop = max(0, y1 - padding)
            new_h = y2 - y1_crop
            new_w = int(new_h / target_ratio)
            cx = x1 + w_box // 2
            x1_crop = max(0, cx - new_w // 2)
            x2_crop = min(w, x1_crop + new_w)
            person_crop = cv_img[y1_crop:y2, x1_crop:x2_crop]
            person_entry = {"box": box}
            preds = []
            if self.upar_model:
                preds.extend(predict_upar(self.upar_model, person_crop, self.upar_active_cols, self.upar_label_idx_map, self.upar_thresholds))
            if self.fashionpedia_model:
                preds.extend(predict_fashionpedia(self.fashionpedia_model, person_crop,
                                                 self.fashionpedia_ordered_ids,
                                                 self.fashionpedia_id2name,
                                                 self.fashionpedia_threshold))
            if preds:
                person_entry["fashion_upar"] = preds
            output["people"].append(person_entry)

        # Face detection + CelebA/FairFace
        face_boxes = []
        if self.face_net is not None:
            face_boxes = detect_faces(self.face_net, cv_img, conf_threshold=FACE_DETECTION_CONF)
        for box in face_boxes:
            x1, y1, x2, y2 = box
            fw = x2 - x1
            fh = y2 - y1
            cx = x1 + fw // 2
            cy = y1 + fh // 2
            side = int(max(fw, fh) * 1)
            cy = max(0, cy - int(side * 0.1))
            sx1 = max(0, cx - side // 2)
            sy1 = max(0, cy - side // 2)
            sx2 = min(w, cx + side // 2)
            sy2 = min(h, cy + side // 2)
            if sy2 <= sy1 or sx2 <= sx1:
                continue  # invalid crop
            pil_face = Image.fromarray(cv2.cvtColor(cv_img[sy1:sy2, sx1:sx2], cv2.COLOR_BGR2RGB))
            face_entry = {"box": box}
            face_preds = []
            if self.fairface_model:
                face_preds.extend(predict_fairface(self.fairface_model, pil_face,
                                                  self.fairface_race_labels,
                                                  self.fairface_gender_labels,
                                                  self.fairface_age_labels,
                                                  self.fairface_thresholds))
            if self.celeba_model:
                face_preds.extend(predict_celeba(self.celeba_model, pil_face,
                                                 self.celeba_attrs_list,
                                                 self.celeba_thresholds))
            if face_preds:
                face_entry["attributes"] = face_preds
            output["faces"].append(face_entry)

        return output


# Singleton instance
pipeline = ModelPipeline()
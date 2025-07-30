# Smart Surveillance Capstone Project

This project demonstrates a multi-model AI pipeline for real-time attribute recognition in surveillance scenarios. It includes:

- **Fashionpedia model**: Detects clothing categories and attributes.
- **UPAR model**: Recognizes person-level visual attributes (e.g. backpack, long hair).
- **FairFace model**: Predicts age, gender, and race from faces.
- **CelebA model**: Recognizes facial appearance attributes (e.g. smiling, eyeglasses).
- **Unified Inference Script**: Combines all models in a single interface with switches to enable/disable specific models.

---

## Directory Tree (excluding image files)

```text
├── Fashionpedia
│   ├── Fashionpedia_old.ipynb
│   ├── data
│   │   ├── annotations
│   │   │   ├── instances_attributes_train2020.json
│   │   │   └── instances_attributes_val2020.json
│   │   ├── test
│   │   └── train
│   └── fashionpedia.ipynb
├── UPAR
│   ├── UPAR.ipynb
│   ├── UPAR_old.ipynb
│   ├── data
│   │   ├── Market1501
│   │   │   ├── Market-1501-v15.09.15
│   │   │   ├── bounding_box_test
│   │   │   ├── bounding_box_train
│   │   │   ├── gt_bbox
│   │   │   ├── gt_query
│   │   │   ├── query
│   │   │   └── readme.txt
│   │   ├── PA100k
│   │   │   ├── data
│   │   │   └── release_data
│   │   │       └── release_data
│   │   ├── PETA
│   │   │   ├── PETA dataset
│   │   │   │   ├── ...
│   │   ├── PETA_ALL_UPAR_labels.csv
│   │   ├── RAP2
│   │   ├── dataset_all.pkl
│   │   └── dataset_all_merged.pkl
│   └── peta_fix.ipynb
├── celeba
│   ├── celeba.ipynb
│   ├── celeba_old.ipynb
│   └── data
│       ├── celeba_cleaned.csv
│       ├── celeba_test.csv
│       ├── celeba_train.csv
│       ├── celeba_val.csv
│       └── ...
├── fairface
│   ├── FairFace.ipynb
│   ├── data
│   │   ├── fairface_label_train.csv
│   │   ├── fairface_label_val.csv
│   │   ├── train
│   │   └── val
│   └── fairface_old.ipynb
├── z_ignore
│   └── ...
├── capstone_project.ipynb
```

---

## How to Use

You can run the unified inference pipeline using the notebook:

```
capstone_project.ipynb
```

Supported input methods:
- Upload a **photo** (e.g. street image, group photo).
- Use your **webcam** for real-time detection and analysis.

Each model can be enabled or disabled with flags in the script:
```python
ENABLE_FAIRFACE = True
ENABLE_CELEBA = True
ENABLE_UPAR = True
ENABLE_FASHIONPEDIA = True
```

---

## Download Trained Models

**Models for this project can be downloaded here:**

👉 [Download models](https://drive.google.com/drive/folders/1-OgwWPJ4Rrz_ahTJiOg9skxvh1Yxl2qC?usp=drive_link)

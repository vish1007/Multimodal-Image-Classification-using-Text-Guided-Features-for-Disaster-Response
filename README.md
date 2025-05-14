
# Multimodal Image Classification using Text-Guided Features for Disaster Response

This project presents a multimodal deep learning pipeline to classify disaster-related social media content using both visual and textual information. The objective is to create effective machine learning classifiers that can assist in real-time disaster response by understanding and labeling images and text from platforms like Twitter.

## 📁 Dataset

We utilize a publicly available multimodal dataset containing annotated tweets from real disaster events. The dataset includes:

* **Images**: Collected from tweets.
* **Text**: Tweet captions and descriptions.
* **Labels**: Human-annotated categories like "Informative", "Non-informative", and various humanitarian purposes.

### Dataset Access

1. **Download** the dataset archive:

  ```bash
   wget https://crisisnlp.qcri.org/data/crisismmd/CrisisMMD_v2.0.tar.gz

   ```

2. **Extract and organize** the files:

   ```bash
   tar -xvf CrisisMMD_v2.0.tar.gz
   mv CrisisMMD_v2.0/data_image $PWD/
   ```

3. **Download pre-trained Word2Vec embeddings**:

   ```
   https://crisisnlp.qcri.org/data/lrec2016/crisisNLP_word2vec_model_v1.2.zip
   ```

   > Place the `.model` file in your working directory and update the model path in `bin/text_cnn_pipeline_unimodal.py`.

---

## 💻 Environment Setup

### Python Environment (Python 2.7)

> Note: This codebase uses Python 2.7. It’s recommended to run it in an isolated environment.

1. **Create a virtual environment**:

   ```bash
   python -m venv multimodal_env
   ```

2. **Activate the environment**:

   ```bash
   source multimodal_env/bin/activate
   ```

3. **Install required dependencies**:

   ```bash
   pip install -r requirements_py2.7.txt
   ```

---

## 🔍 Running Unimodal Classifiers

### Text-Only Classification

Train a CNN using tweet texts:

```bash
CUDA_VISIBLE_DEVICES=0 python bin/text_cnn_pipeline_unimodal.py \
-i data/task_data/task_informative_text_img_agreed_lab_train.tsv \
-v data/task_data/task_informative_text_img_agreed_lab_dev.tsv \
-t data/task_data/task_informative_text_img_agreed_lab_test.tsv \
--log_file snapshots/informativeness_cnn_keras.txt \
--w2v_checkpoint w2v_checkpoint/word_emb_informative_keras.model \
-m models/informativeness_cnn_keras.model \
-l labeled/informativeness_labeled_cnn.tsv \
-o results/informativeness_results_cnn.txt \
>& log/text_info_cnn.txt &
```

### Image-Only Classification

Train a CNN using VGG16 features extracted from images:

```bash
CUDA_VISIBLE_DEVICES=0 python bin/image_vgg16_pipeline.py \
-i data/task_data/task_informative_text_img_agreed_lab_train.tsv \
-v data/task_data/task_informative_text_img_agreed_lab_dev.tsv \
-t data/task_data/task_informative_text_img_agreed_lab_test.tsv \
-m models/informative_image.model \
-o results/informative_image_results_cnn_keras.txt \
>& log/informative_img_vgg16.log &
```

---

## 🔗 Running Multimodal Classification (Text + Image)

### Step 1: Preprocess images into NumPy format

```bash
python bin/image_data_converter.py -i data/all_images_path.txt -o data/task_data/all_images_data_dump.npy
```

### Step 2: Train and test a combined model

```bash
CUDA_VISIBLE_DEVICES=1 python bin/text_image_multimodal_combined_vgg16.py \
-i data/task_data/task_informative_text_img_agreed_lab_train.tsv \
-v data/task_data/task_informative_text_img_agreed_lab_dev.tsv \
-t data/task_data/task_informative_text_img_agreed_lab_test.tsv \
-m models/info_multimodal_combined.model \
-o results/info_multimodal_results.txt \
--w2v_checkpoint w2v_checkpoint/data_w2v_info_combined.model \
--label_index 6 \
>& log/info_multimodal.log &
```

---

## 📂 Project Structure

```
multimodal_disaster_response/
│
├── bin/                        # All scripts for training and evaluation
│   ├── text_cnn_pipeline_unimodal.py
│   ├── image_vgg16_pipeline.py
│   ├── text_image_multimodal_combined_vgg16.py
│   └── image_data_converter.py
│
├── data/                      # Dataset and processed files
│
├── models/                    # Saved models
│
├── results/                   # Output predictions
│
├── labeled/                   # Annotated predictions
│
├── log/                       # Log files for debugging
│
├── requirements_py2.7.txt     # Python 2.7 dependencies
└── README.md
```

---

## ⚠️ Notes

* Python 2.7 is deprecated. For long-term maintenance or custom extension, consider porting the codebase to Python 3.x.
* You can modify the scripts to use newer models like BERT for text or ResNet for image features.

---

## 🧠 Contributions

This project is based on the study of disaster-related multimodal classification. Contributions to enhance the model with newer architectures or cleaner preprocessing pipelines are welcome.

---

## 📜 License

This project is for research and educational use only. Please refer to the original dataset license terms on [CrisisNLP](https://crisisnlp.qcri.org/crisismmd.html).

---

## 🙏 Acknowledgements

We appreciate the efforts of the research community in curating and publishing high-quality datasets for disaster response. This work builds on publicly available resources with deep gratitude.

---


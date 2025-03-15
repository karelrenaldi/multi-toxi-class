# Multilingual Toxic Classification

This project implements a **multilingual toxic comment classification** system for **English and Russian** text. It consists of three main pipelines:

1. **Language Detection** – Determines whether the input text is in English or Russian.
2. **Machine Translation** – Translates Russian text to English (if necessary).
3. **Toxicity Classification** – Classifies the translated (or original) text for toxicity using pre-trained models.

## Datasets
- **TensorFlow `ted_hrlr_translate/ru_to_en`** – Used for machine translation.
- **Hugging Face toxicity classification datasets** – Used for training the toxicity detection model.

## Results & Limitations
While the pipeline successfully detects toxic comments in **English**, it struggles with **Russian** text due to translation limitations. Key issues include:
- **Machine translation inaccuracies**, leading to loss of meaning.
- **Out-of-Vocabulary (OOV) issues**, as each model is trained on different datasets.
- **Insufficient training data**, affecting overall performance.

## Future Improvements
- Enhance machine translation quality for better context preservation.
- Fine-tune toxicity classification models on multilingual datasets.
- Investigate alternative approaches for Russian text classification without translation.

## YT Videos
https://www.youtube.com/watch?v=MniHS6C7zEo

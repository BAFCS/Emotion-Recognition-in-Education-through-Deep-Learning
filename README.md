# Emotion Recognition in Education through Deep Learning

This repository contains the code developed for the thesis at Universidad Andrés Bello, aimed at creating innovative unimodal facial and textual models for emotion recognition in the educational field, with the purpose of providing robust and accurate models applicable to educational systems, facilitating feedback and comprehension of emotions, and laying the foundation for future multimodal integration.

**Pretrained Models**

This repository includes the full codebase for the project. The pretrained models are available on Hugging Face and can be accessed at the following link:
[https://huggingface.co/BAFCS/Emotion-Recognition-in-Education-through-Deep-Learning](https://huggingface.co/BAFCS/Emotion-Recognition-in-Education-through-Deep-Learning)

The models are stored in the "Code" folder. For the facial modality, there are four codes: two implementing individual models, each using a different data balancing method, and two named "Ensemble," which present ensemble models developed by combining different architectures. Among these, the file "2_codeEnsembleWithWeights.ipynb" stands out as the most relevant. Regarding the textual modality, only the file "code.ipynb" is included, which uses the CSV files generated directly within the same code.

For the facial modality, a methodology was proposed that combines two widely used datasets in emotion recognition: FER2013 and CK+48. This combination was complemented by two data balancing approaches: the use of weighted weights to adjust the importance of each class according to its representation in the dataset, and the generation of synthetic data using data augmentation techniques, which helped balance the classes. The model developed for this modality is an ensemble model, designed to improve generalization in emotion recognition.
![MetodologiaFacial3](https://github.com/user-attachments/assets/0c25c728-8492-44f5-bb85-2ecff14e22ff)

Regarding the textual modality, two methodologies were developed based on emotions commonly experienced in educational settings. The first focuses on a set of five core emotions typically found in educational contexts, while the second expands this set by incorporating two additional emotions: "Surprise" and "Neutral", which represent emotional states observed in students and are supported by relevant scientific literature. For the training set, data augmentation techniques were applied to enhance its diversity, along with thorough data cleaning to ensure quality. Subsequently, the data were transformed using embeddings and pre-trained models, with RoBERTa combined with an N-Gram architecture standing out for its ability to capture key features for emotion recognition.
![MetodologiaTextual](https://github.com/user-attachments/assets/e80f01e3-b491-4233-a8b6-2a41a7f8cb55)


- Facial modality results:

| Model FER2013                                                                                       |  Accuracy | Precision | Recall | F1-Score |
|---------------------------------------------------------------------------------------------|------------------|-----------|--------|----------|
| Model 1 ensemble with CNN-1, VGG19, CNN-6 (weight balancing)                                   | 68.15%           | 67%       | 67%    | 66%      |
| Model 2 ensemble with CNN-1, CNN-2, VGG19 (data balancing)                                     | 69.61%           | 71%       | 67%    | 68%      |
| Model 3 ensemble with CNN-1, CNN-2 (first augmentation), CNN-2 (second augmentation) (weight balancing) and CNN-1, VGG19 (data balancing) | 69.94%           | 69%       | 68%    | 68%      |
| Model 4 ensemble with CNN-1, CNN-2 (first augmentation), CNN-2 (second augmentation) (weight balancing) and CNN-1, CNN-2, VGG19 (data balancing) | 70.20%           | 70%       | 68%    | 69%      |

| Model FER2013 + CK+48                                                                                       |  Accuracy | Precision | Recall | F1-Score |
|---------------------------------------------------------------------------------------------|--------------------------|-----------|--------|----------|
| Model 1 ensemble with CNN-1, CNN-2, CNN-3 (weight balancing) and CNN-2 (data balancing)         | 68.65%                   | 67%       | 68%    | 67%      |
| Model 2 ensemble with CNN-1, CNN-2, Keras-Tuner (weight balancing) and CNN-1, Keras-Tuner (data balancing) | 69.82%                   | 69%       | 69%    | 69%      |
| Model 3 ensemble with CNN-1, CNN-2, Keras-Tuner (weight balancing) and CNN-1, Keras-Tuner, CNN-2 (data balancing) | 69.90%                   | 70%       | 69%    | 69%      |
| Model 4 ensemble with CNN-1, CNN-2, CNN-3 (weight balancing) and CNN-1, VGG19 (data balancing)  | 70.13%                   | 70%       | 69%    | 69%      |
| Model 5 ensemble with CNN-1, CNN-2, CNN-3 (weight balancing) and CNN-1, CNN-2, VGG19 (data balancing) | 70.26%                   | 70%       | 69%    | 69%      |
| Model 6 ensemble with CNN-1, CNN-2, Keras-Tuner (weight balancing) and CNN-1, CNN-2, VGG19 (data balancing) | 70.36%                   | 70%       | 69%    | 70%      |



- Textual modality results:

| Models 5 Emotions                                                                                     |  Accuracy | Precision | Recall | F1-Score |
|--------------------------------------------------------------------------------------------|---------------------|-----------|--------|----------|
| Model 3 Word2Vec-CNN 128 filters - 2,3,4 kernel - 128 batch                                          | 71.53%              | 72%       | 72%    | 71%      |
| Model 4B ERT-CNN 128 filters - 2,3,4 kernel - 128 batch- 1e-4 lr                                       | 78.10%              | 78%       | 78%    | 78%      |
| Model 16 BERT-CNN 128 filters - 2,3,4 kernel - 64 batch- 1e-4 lr                                       | 78.28%              | 78%       | 78%    | 78%      |
| Model 20 RoBERTa-CNN 128 filters - 2,3,4 kernel - 128 batch- 1e-4 lr                                   | 78.64%              | 79%       | 79%    | 79%      |

| Models 7 Emotions                                                                                     |  Accuracy | Precision | Recall | F1-Score |
|--------------------------------------------------------------------------------------------|---------------------|-----------|--------|----------|
| Model 9 ALBERT-CNN 128 filters - 2,3,4 kernel - 128 batch - 1e-4 lr - doble clean                          | 72.75%              | 73%       | 73%    | 73%      |
| Model 8 XLNet-CNN 128 filters - 2,3,4 kernel - 128 batch - 1e-4 lr - doble clean                           | 72.88%              | 73%       | 73%    | 73%      |
| Model 7 DistilBERT-CNN 128 filters - 2,3,4 kernel - 128 batch - 1e-4 lr - doble clean                      | 79.79%              | 80%       | 80%    | 80%      |
| Model 10 DeBERTa-CNN 128 filters - 2,3,4 kernel - 128 batch - 1e-4 lr - doble clean                        | 80.18%              | 80%       | 80%    | 80%      |
| Model 4 BERT-CNN 128 filters - 2,3,4 kernel - 128 batch - 1e-4 lr                                       | 80.18%              | 80%       | 80%    | 80%      |
| Model 6 RoBERTa-CNN 128 filters - 2,3,4 kernel - 128 batch - 1e-4 lr - doble clean                         | 80.70%              | 81%       | 81%    | 81%      |








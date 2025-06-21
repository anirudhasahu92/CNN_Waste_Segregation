# Waste Material Segregation for Improving Waste Management

> This project implements a Convolutional Neural Network (CNN) based system to classify waste materials into different categories to improve waste management efficiency and sustainability.

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)
<!-- You can include any other section that is pertinent to your problem -->

## General Information
- What is the background of the project?<br>
	- Effective waste segregation is a critical step in recycling and reducing landfill waste. This project aims to automate this process using computer vision.
- What is the business probem that your project is trying to solve?<br>
	- The project addresses the challenge of accurately identifying and sorting different types of waste materials automatically, which is currently a labor-intensive and error-prone manual process.
- What is the dataset that is being used?
    - The dataset consists of images of common waste materials categorized into several classes: Cardboard, Food_Waste, Glass, Metal, Paper, and Plastic.
- Project Steps
1. **Data Loading and Exploration:**
    - Loaded and unzipped the dataset. Examined the directory structure and counted the number of images per class to understand the distribution.
2. **Data Preprocessing:**
    - Initial data loading using `tf.keras.utils.image_dataset_from_directory` revealed severe class imbalance and an unrepresentative validation set.
    - Implemented a manual data splitting process using `train_test_split` with `stratify` to ensure a balanced distribution of classes in both training and validation sets.
    - Resized images to a fixed size (256x256) and normalized pixel values to the range [0, 1].
    - Explored data augmentation techniques but found they did not improve performance with the chosen parameters.
3. **Model Building:**
    - Built several CNN models with three convolutional layers, followed by MaxPooling, Batch Normalization, Dropout, Flattening, and Dense layers.
    - Experimented with different configurations, including adding L2 regularization and adjusting learning rates.
4. **Model Evaluation:**
    - Evaluated models on the stratified validation dataset using accuracy and loss metrics.
    - Generated classification reports to assess performance on individual classes.
    - Evaluated models on unseen sample images to test generalization ability.

## Conclusions
1. **Data Issues are Critical:** The dataset suffers from significant within-class variability, making it hard for the model to identify distinguishing features for each category. Additionally, the initial data loading method created a severely imbalanced and unrepresentative validation set, rendering early model evaluations meaningless.
2. **Manual Stratification was Necessary but Insufficient:** Manually splitting the data with stratification corrected the class imbalance issue in the validation set. However, the inherent variability within classes and the limited number of images per class in the training set hampered the model's ability to generalize, leading to only moderate accuracy.
3. **Augmentation was Not Effective:** Data augmentation, with the chosen parameters, did not improve performance. The high within-class variability likely caused the augmented images to further confuse the model.
4. **Model Performance is Limited:** The trained CNN models, even with regularization, achieved only moderate accuracy (around 60-70%) on unseen data. The high variability within classes is a major limiting factor for this simple CNN architecture.
5. **Future Work is Required:** To achieve higher accuracy and better generalization, future efforts must focus on:
    - Utilizing a larger and more diverse dataset with less within-class variability.
    - Exploring transfer learning with pre-trained CNN models.
    - Implementing more sophisticated and carefully tuned data augmentation techniques.
    - Performing extensive hyperparameter tuning and exploring more complex model architectures.
6. **The first model performed better on unseen data:** Although the initial validation process for the first model was flawed due to the biased validation dataset (dominated by a single class), its performance on a separately sampled set of unseen images was surprisingly better than other models. This suggests that while the subsequent models trained on the stratified data had more reliable validation metrics, the initial model, despite its validation issues, learned features from the original (unstratified) training distribution that generalized slightly better to varied unseen examples. This highlights that validation performance on a poorly representative dataset can be misleading and that evaluation on truly independent and diverse unseen data is crucial for assessing generalization. It also implies that the later models, while trained on a balanced dataset, might have overfitted to the nuances introduced by the manual stratification and potentially reduced per-class training sample counts compared to the original distribution.

## Technologies Used
- Python (3.x)
- TensorFlow / Keras (2.x)
- NumPy
- Pandas
- Seaborn
- Matplotlib
- PIL (Pillow)
- OpenCV (cv2)
- Scikit-learn (sklearn)
- OS
- Random
- Glob
- Shutil

## Acknowledgements
I would like to thank Upgrad for giving me this opportunity to work on this project.

## Contact
Created by [Anirudha Kumar Sahu](https://github.com/anirudhasahu92) - feel free to contact me!

<!-- Optional -->
<!-- ## License -->
<!-- This project is open source and available under the [... License](). -->

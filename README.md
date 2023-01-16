# ELEC0134 Final Assignment 2022-2023
## Project Organization
The project is organized in the following manner:
- Folders A1, A2, B1 and B2 contain the python scripts that perform the data pre-processing, training, and evaluating of the different tasks (**task_a1.py, task_a2.py, task_b1.py, task_b2.py**). Each of those files contains an evaluator class.
- The **main.py** file will be in charge of running the different tasks, by calling the different evaluator classes. In order to run a specific task, the following command must be input:
```console
python main.py -t task_to_be_run
```
Note: The task_to_be_run variable must be one of the following: task_a1, task_a2, task_a3, task_a4.
- The **facial_regions_detector.py** file contains the same functions that were given to us in the file lab3_data.py file for the lab 7 notebook. Only one of the functions is slightly modified. It can be used (alongside the **shape_predictor_68_face_landmarks.dat** file) to extract the facial landmarks from images.
- The **Datasets** folder is used to copy-paste the datasets into it in order to run the evaluations.
- The remaining **jupyter notebooks** were utilized to track the progress. However, in order for the evaluation of the models, they are not necessary. They should be ignored.

## Packages required
- numpy
- scikit-learn (sklearn)
- pandas
- pillow (PIL)
- opencv-python (cv2)
- tensorflow
- dlib
# Face-Detection-Using-Resnet50

1.	Introduction
This Flask-based web application aims to detect faces in uploaded images and perform similarity comparison and eye distance measurement between them. The app utilizes OpenCV, MTCNN (Multi-task Cascaded Convolutional Networks), and ResNet-50 for face detection, feature extraction, and similarity calculation.

2. Prerequisites
- Python 3.x
- Flask
- OpenCV
- MTCNN
- TensorFlow (for ResNet-50)
- Other required libraries (numpy, pandas, matplotlib, seaborn)
3. Front End

Page Overview:
This user interface is designed to work with a server-side application for image uploading and processing. It includes sections for uploading images, displaying uploaded images, showing similarity results, presenting eye distance results, and displaying a confusion matrix.
Image Upload Form: Allows users to upload two image files. Images are restricted to be of the image type and must be selected for both fields before submission.

Uploaded Images: Displays uploaded images if available. Uses conditional checks to show images.

Similarity Result: Displays a similarity result if available.

Eye Distance Result: Displays eye distance results if available. Includes an image detecting both eyes and eye distance measurement.
Confusion Matrix: Displays a confusion matrix if available. Presented in a table format, indicating predicted and actual outcomes.

4.  Initialization and Setup
The `FaceDetectionApp` class initializes the Flask app, configures necessary parameters, and sets up routes.

#### `__init__(self)`
- Initializes the Flask app, file upload folder, paths, and detectors.
- Defines parameters like similarity threshold.
- Initializes variables to store results and paths.

#### `_setup_routes(self)`
- Sets up the app route for handling requests.

#### `run(self)`
- Starts the Flask app for debugging.

### Index Page
The main page of the app where users can upload images and see results.

#### `index(self)`
- Handles POST requests for uploaded images.
- Processes images, detects faces, calculates similarity, and measures eye distance.
- Returns the rendered HTML template with results.

5. Image Processing

Functions to process uploaded images and detect faces.

#### `_process_uploaded_images(self, uploaded_image1, uploaded_image2)`
- Saves uploaded images to the server and returns their paths.

#### `_detect_id_card_face(self, image_path, output_path)`
- Detects ID card face using SIFT feature matching and homography.
- Saves the detected ID card image and detected faces.

#### `_detect_real_image_face(self, image_path)`
- Detects faces in the uploaded real image.

### Similarity Calculation
Functions to calculate similarity between detected faces.

#### `_calculate_similarity(self, id_card_face, real_image_face)`
- Preprocesses images and calculates cosine similarity using ResNet-50 features.
- Compares similarity score to a threshold and returns the result.

### Eye Distance Measurement
Functions to measure the distance between eyes.

#### `_calculate_eye_distance(self, image_path)`
- Detects faces and eyes in the image.
- Calculates the distance between the detected eyes in millimeters.

Face Extraction
Function to extract faces from images.

#### `extract_face(self, image_path, required_size=(224, 224))`
- Uses MTCNN to detect faces and extract them from the image.

### Flask App Initialization
Entry point to run the Flask app.

#### `if __name__ == '__main__':`
- Creates an instance of `FaceDetectionApp` and starts the app.

6. Conclusion
This Flask-based web application demonstrates face detection, similarity calculation, and eye distance measurement using OpenCV, MTCNN, and ResNet-50. Users can upload images, and the app provides results related to face similarity and eye distance. It's a useful tool for scenarios such as ID card verification or security applications.

# Mask Detection Model and Web App

This repository contains a Python script to train a mask detection model and a web application to test if a person is wearing a mask or not.

## Trained Model

### Saving the Trained Model

To train and save the model, follow these steps:

1. **Install Dependencies:** Ensure you have the necessary dependencies installed. You can install them using pip:

    ```bash
    pip install -r requirements.txt
    ```

2. **Train and Save Model:** Run the `main.py` script to train the model and save it to the root folder:

    ```bash
    python main.py
    ```

    The trained model will be saved as `mask-detection-model.h5`.

## Web Application

### Running the Web Application

To run the web application, follow these steps:

1. **Install Dependencies:** Install the required dependencies using pip:

    ```bash
    pip install -r requirements.txt
    ```

2. **Start the Web App:** Start the Flask web application by running the `app.py` script:

    ```bash
    python app.py
    ```

3. **Access the Web App:** Open a web browser and go to [http://localhost:5000](http://localhost:5000) to access the web application.

### Using the Web Application

1. **Upload an Image:** Upload an image of a person to the web application.

2. **Detect Mask:** Click the "Upload and Detect" button.

3. **View Results:** The application will process the image and display the result, indicating whether the person in the image is wearing a mask or not.

## File Structure

- `main.py`: Python script to train the mask detection model and save it.
- `app.py`: Python script for the Flask web application.
- `data/`: Directory containing the dataset of images with and without masks.
- `templates/`: Directory containing HTML templates for the web application.
- `static/`: Directory for static files used in the web application.
- `mask-detection-model.h5`: The trained model file.

## Dependencies

- Python 3.x
- TensorFlow
- Keras
- Flask
- Pillow (PIL)
- scikit-learn

## Dataset

The dataset used for training the model is located in the `data/` directory. It includes images of people with and without masks.

## License

This project is licensed under the [MIT License](LICENSE). As an academic project, you are encouraged to adapt and use this code for educational purposes. Please check the [LICENSE](LICENSE) file for full details.

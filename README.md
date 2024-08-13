# Sign Language Interpreter

This project aims to develop an **Indian Sign Language Interpreter** using Python, MediaPipe, and TensorFlow. It leverages computer vision and machine learning to recognise and translate sign language gestures into text.

## Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Collection](#data-collection)
- [Model Training](#model-training)
- [Testing](#testing)
- [Configuration](#configuration)
- [License](#license)
- [Contact](#contact)

## Project Structure

```plaintext
sign_language_interpreter/
├── config/
│   └── config.py
├── data/
│   └── raw/
│       └── (raw data files)
├── dataset/
│   └── (captured gesture data)
├── logs/
│   └── (log files)
├── models/
│   └── (saved models)
├── notebooks/
│   └── (Jupyter notebooks for experiments and analysis)
├── src/
│   ├── __init__.py
│   ├── data_collection.py
│   ├── model.py
│   ├── test.py
│   ├── utils.py
├── labels.txt
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/ACM40960/project-bhupendrachaudhary08.git
   ```

2. **Create a virtual environment:**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Data Collection

To collect data for sign language gestures, you can use the `data_collection.py` script. The collected data will be stored in the `dataset/` directory.

1. **Prepare your labels:**
   Create a `labels.txt` file with each line containing a label for a gesture.

2. **Run the data collection script:**
   ```sh
   python src/data_collection.py
   ```

   Follow the on-screen instructions to capture gesture data for each label.

## Model Training

To train the model, you can use the `model.py` script. The training process will save the trained model in the `models/` directory and logs in the `logs/` directory.

1. **Run the model training script:**
   ```sh
   python src/model.py
   ```

2. **Monitor the training process:**
   You can use TensorBoard to visualise the training process:
   ```sh
   tensorboard --logdir logs
   ```

## Testing

To test the trained model with live video feed, you can use the `test.py` script.

1. **Run the test script:**
   ```sh
   python src/test.py
   ```

   The script will capture video from your webcam, process the frames, and display the predicted gestures.

## Configuration

All configurable parameters are located in the `config/config.py` file. This includes paths, data collection settings, model settings, and logging settings.

## License

This project is licensed under the MIT License.

## Contact

For any questions or suggestions, please open an issue or contact me at [bhupendra.chaudhary@ucdconnect.ie](mailto:bhupendra.chaudhary@ucdconnect.ie).

## Credits

This project is in collaboration with [Sahil Chalkhure](https://github.com/ACM40960/project-sahilchalkhure26)
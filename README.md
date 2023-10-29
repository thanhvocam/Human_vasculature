# Blood Vessel Segmentation Project

## Description
This project aims to build an automated system for segmenting blood vessels in scanned images. Blood vessel segmentation is the process of classifying and marking blood vessel regions in an image, separating them from other structures.

## Example with a image after predict with model.
![Before predict](before_pred.png)
![After predict](after_pred.png)

## Optional
1. If you want to use Docker for deployment, install Docker using the Dockerfile: `docker build -t blood-vessel-segmentation .`
2. If you want to use Streamlit app access: `https://humanvasculature-fehopwpmus7frpakvv2ngt.streamlit.app/`
3. If you want to try to see the prediction results immediately, you can run the `infer.py`

## Installation
1. Install the Python environment using the `env.yaml` file. Run the following command to install: `conda env create -f env.yaml`
2. Activate the installed environment: `conda activate human_vasculature`
3. Install the required packages: `pip install -r requirements.txt`

## Usage
- Train the model: Run the `train_model.py` file to train the segmentation model on the training data.
- Predict and segment blood vessels on new images: Run the `infer.py` file to predict and segment blood vessels on new images. The path to the image to be predicted is specified in the `infer.py` file.
- Run the Streamlit application: Run the `streamlit_app.py` file to start the Streamlit application to display and interact with the blood vessel segmentation results on a web interface.

## Directory Structure
- `best.pt`: The weights of the best-trained model.
- `Dockerfile`: Configuration file for deploying the application using Docker.
- `env.yaml`: Environment file to create a conda environment.
- `example.py`: Example file containing functional code.
- `infer.py`: File containing functional code to predict and segment blood vessels on new images.
- `packages.txt`: List of required packages.
- `requirements.txt`: List of required packages and their versions.
- `streamlit_app.py`: File containing functional code to run the Streamlit application.
- `train_model.py`: File containing functional code to train the segmentation model.

## Contributing
If you would like to contribute to this project, please create a pull request and wait for feedback from the development team.

## Author
- Name: [Thanhvocam](https://github.com/thanhvocam/Human_vasculature.git)
- Email: thanhvocam94@gmail.com

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
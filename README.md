# Mask Detection
A simple real-time mask-detection program capable of detecting masks on multiple faces through webcam input.

# Sample
![](sample.gif)

# Setup
Install dependencies
'''sh
apt update
apt install git python3-dev python3-pip build-essential libagg-dev pkg-config
'''

Clone repository
```sh
git clone https://github.com/damianomarsili/mask-detection.git
cd mask-detection
```

Install requirements
```sh
python3 -m venv env
. env/bin/activate
pip3 install -r requirements.txt
```

# Program Execution
To run the program, run mask_detection.py:
```sh
python3 mask_detection.py
```

To exit the program, select the image frame and hit esc. Or, alternatively, select the terminal and type ctrl + c and enter.

# Custom Training (Optional)
To train the network yourself, create a directory named `data`. Inside this directory, create two directories `train` and `valid`. Within both of these directories, create two more directories named `mask` and `no-mask` respectively. Include the images to use for training within their respective directories.

You will also need to install the pandas library:
```sh
pip3 install pandas
```
You can train the model by running train.py:
```sh
python3 train.py
```

# Resources:
- Mediapipe documentation: https://github.com/google/mediapipe
- Mask detection dataset: https://www.kaggle.com/prithwirajmitra/covid-face-mask-detection-dataset 

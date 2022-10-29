# Auto Traffic Congestion
# Setup:

## Windows
```bash
py -m pip install --upgrade pip
py -m pip install --user virtualenv
py -m venv env
.\env\Scripts\activate
pip install -r requirements.txt
py thesis.py --output_folder output_folder --input input_video
```

## MacOS
```bash
python3 -m pip install --upgrade pip3
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip3 install -r requirements.txt
python3 thesis.py --output_folder output_folder --input input_video
```

## Goolge Colab/ Jupiter Notebook
- setup cv2 custom to use gpu: https://towardsdatascience.com/how-to-use-opencv-with-gpu-on-colab-25594379945f
```python 
!sudo apt install python3-dev python3-pip python3-testresources
!sudo apt install build-essential cmake pkg-config unzip yasm git checkinstall
!sudo apt install libjpeg-dev libpng-dev libtiff-dev
!sudo apt install libavcodec-dev libavformat-dev libswscale-dev libavresample-dev
!sudo apt install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev
!sudo apt install libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev
!sudo apt install libfaac-dev libmp3lame-dev libvorbis-dev
!sudo apt install libopencore-amrnb-dev libopencore-amrwb-dev
!sudo apt-get install libgtk-3-dev
!sudo apt-get install libtbb-dev
!sudo apt-get install libatlas-base-dev gfortran
!sudo apt-get install libprotobuf-dev protobuf-compiler
!sudo apt-get install libgoogle-glog-dev libgflags-dev
!sudo apt-get install libgphoto2-dev libeigen3-dev libhdf5-dev doxygen
!python3 thesis.py --output_folder output_folder --input input_video
```
![image](https://user-images.githubusercontent.com/49317519/127734879-8e2321c8-16ea-4466-b4ee-4279f01b1fed.png)

# extrinsics_calibr

The current multi-camera external parameter calibration program is full of bugs due to frequent API updates by opencv. What's worse, since LLM models such as Chatgpt are trained on old data, the generated calibration code can't adapt to the API of the current openCV library.This library is a multi-camera calibration program based on the latest opencv version, which was indignantly written from scratch by the author after he failed to try several open-source solutions.

Features are as follows: supports the latest opencv library, no need to modify any API manually. only supports realsense calibration. Other cameras need to get the internal reference first before you can.
## Step0: prepare your board
```
python step0_make_board.py --board board.yaml
```

## Step1: extract image from video
```
python step1_extractor.py --basecam_id 2 --subcam_id 1 --task_path ./calibration
python step1_extractor.py --basecam_id 2 --subcam_id 3 --task_path ./calibration
python step1_extractor.py --basecam_id 2 --subcam_id 4 --task_path ./calibration
```

## Step2: using image to calibrate extrinsic parameters
```
python step2_calibrator.py --board board.yaml --basecam_id 2 --subcam_id 1 --task_path ./calibration
python step2_calibrator.py --board board.yaml --basecam_id 2 --subcam_id 3 --task_path ./calibration
python step2_calibrator.py --board board.yaml --basecam_id 2 --subcam_id 4 --task_path ./calibration
```

## Step3: visualize the result
```
python step3_visualize.py --basecam_id 2 --subcam_id 1 3 4 --task_path ./calibration
```

## file structure
```
easy_camcali/
    ├── step0_make_board.py   # make calibration board
    ├── step1_extractor.py    # extract image from stream
    ├── step2_calibrator.py   # loading image and intrinsics to calibrate extrinsic parameters
    ├── step3_visualize.py    # visualize the calibration result
    ├── board.yaml      # board configuration
    ├── readme.md       # description and usage
    ├── calibration     # store the data and result
    │   ├── cali_T_1_2      # calibration task1
    │   │   |-- cam1        # image from cam1
    │   │   |   |-- 0001.jpg
    │   │   |   |-- ...
    │   │   |-- cam2    # image from cam2
    │   │   |   |-- 0001.jpg
    │   │   |   |-- ...
    |   |   |-- intrinsics.yaml # intrinsics parameters
    │   │   |-- extrinsics.yaml     # extrinsics parameters
    │   │
    |   ├── cali_2_3    # calibration task2
    │   │
```
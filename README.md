# extrinsics_calibr

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
# 1. 数据集

[Bosch Small Traffic Lights Dataset](https://aistudio.baidu.com/datasetdetail/255499)

# 2. 指标

| model | config | mAP | precision | recall | RedLeft | Red | RedRight | GreenLeft | Green | GreenRight | Yellow | off |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| YOLOv3 | [config](../configs/bosch_40ep_yolov3_fpn_dbsampler_warmup.yaml) | 29.40 | 25.25 | 38.14 | 44.65 | 47.98 | 0.00 | 30.63 | 68.39 | 0.00 | 25.13 | 18.40 |
| YOLOv3 | [config](../configs/bosch_40ep_yolov3_tiny_fpn_cbgs_dbsample_warmup.yaml) | 33.12 | 34.42 | 40.82 | 48.42 | 53.11 | 0.00 | 45.39 | 72.76 | 0.00 | 32.87 | 12.44 |

# MS-DETR
This is the official repository of work:MS-DETR: Multispectral Pedestrian Detection Transformer with Loosely Coupled Fusion and Modality-Balanced Optimization.

## Build && Update
### 
  - 2023/7/21 Build the official repository of our MS_DETR and upload the evalution scripts and the detection results of our MS-DETR and other sota multispectral detectors on the KAIST dataset. We will update all codes and models after our work is accepted.

## Evalutation_script

You can evaluate the result files of the models with code.

We draw all the results of state-of-the-art methods in a single figure to make it easy to compare, and the figure represents the miss-rate against false positives per image.

For annotations file, only json is supported, and for result files, json and txt formats are supported.
(multiple `--rstFiles` are supported)

Example

```bash
$ python evaluation_script.py \
	--annFile KAIST_annotation.json \
	--rstFile state_of_arts/ACF_result.txt \
            state_of_arts/ARCNN_result.txt \
            state_of_arts/CIAN_result.txt \
            state_of_arts/Fusion-RPN+BF_result.txt \
            state_of_arts/Halfway-Fusion_result.txt \
            state_of_arts/IAF-RCNN_result.txt \
            state_of_arts/IATDNN-IAMSS_result.txt \
            state_of_arts/MBNet_result.txt \
            state_of_arts/GAFF_result.txt \
            state_of_arts/MLPD_result.txt \
            state_of_arts/MSDS-RCNN_result.txt 
```
![result img](./evaluation_script/FPPI_Reasonable.jpg)
![result img](./evaluation_script/FPPI_ALL.jpg)

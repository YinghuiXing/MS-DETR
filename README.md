# MS-DETR
This is the official repository of work:MS-DETR: Multispectral Pedestrian Detection Transformer with Loosely Coupled Fusion and Modality-Balanced Optimization.

## Build && Update
### 
  - 2024/7/12 Release our MS-DETR code. **At the same time we are focusing on developing RGBT models that are more efficient than MS-DETR. Stay tuned.**
  - 2023/7/21 Build the official repository of our MS-DETR and upload the evalution scripts and the detection results of our MS-DETR and other sota multispectral detectors on the KAIST dataset. We will update all codes and models after our work is accepted.

## Installation guide

To install the necessary dependencies for this project, please follow the steps below:

1. **Install Dependencies**:
   Navigate to the directory containing the `requirements.txt` file and run the following command to install all required packages:

   ```bash
   conda install --file ./requirements.txt
   ```

2. **Ensure Proper Installation of MSDA Operator**:
   After installing the dependencies, ensure the MSDA operator is correctly installed by navigating to the `exp/official_repo/MS-DETR/models/dab_deformable_detr/ops` directory:

   ```bash
   cd exp/official_repo/MS-DETR/models/dab_deformable_detr/ops
   bash make.sh  # install 
   ```

## How to train MS-DETR?
1. Pretrain model link [GoogleDisk](https://drive.google.com/file/d/10kCkBytXbp5Ke-xqWpYLIZ3oWwVQi0Gp/view?usp=sharing). Please download them and place them in the pretrain_models directory.
2. Please refer to the exp_config folder for more details. There are training and test script commands in each yaml file. Take KAIST training as an example.
```bash 
torchrun --nproc_per_node=4 --master_port=49104 fusion_main.py --exp_config exp_config/KAIST/kaist.yaml --output_dir <path of your work dir> --action train
```

## How to evaluate MS-DETR?
```bash 
python fusion_main.py --output_dir <output dir> --action test --resume <path of checkpoint.pth> --exp_config <path of exp config>
```

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
![result img](evaluation_script/FPPI_Reasonable.jpg)
![result img](evaluation_script/FPPI_All.jpg)

## Citation

If you find this code helpful, please kindly cite:
```bib
@article{xing2023multispectral,
  title={Multispectral Pedestrian Detection via Reference Box Constrained Cross Attention and Modality Balanced Optimization},
  author={Xing, Yinghui and Wang, Song and Liang, Guoqiang and Li, Qingyi and Zhang, Xiuwei and Zhang, Shizhou and Zhang, Yanning},
  journal={arXiv preprint arXiv:2302.00290},
  year={2023}
}
```
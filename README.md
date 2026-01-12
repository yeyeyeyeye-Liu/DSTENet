# DSTENet
This repository provides the implementation of DSTENet, a deep learning framework integrating Deformable RoI Pooling, ConvNeXt, and Transformer architectures for identifying hazard-prone areas of gully-type debris flows.


The code corresponds to the following paper:

A Deformable RoI Pooling and ConvNeXt–Transformer Integrated Approach for Identifying Hazard-Prone Areas of Gully-Type Debris Flows: A Case Study in Typical Regions of Yunnan, China

under review at *Computers & Geosciences*.

--------

1. Environment

- Python 3.10  
- PyTorch 2.4.0  

The main dependencies are listed in `requirements.txt`.

--------

2. Installation

The required Python dependencies can be installed using:

```bash
pip install -r requirements.txt
```

--------

3. Usage

The implementation of the DSTENet model, as well as the training and testing procedures, are integrated into a single script: DSTENet.py.

Training

Model training can be started by running:
python DSTENet.py --batch_size 4 --max_epoch 200 --lr 0.0005


Key training parameters include:
	•	--batch_size: batch size for training (default: 4)
	•	--max_epoch: number of training epochs (default: 200)
	•	--lr: learning rate (default: 0.0005)
	•	--num_workers: number of data loading workers (default: 2)
	•	--class_num: number of output classes (default: 8)

Testing / Evaluation

The same script is used for model evaluation, depending on the configuration and checkpoint loading strategy defined in the code.

--------

4. Data Availability and Format

Data Availability

The dataset used in this study is not fully publicly available due to regional data-sharing restrictions.
However, a representative example data file is provided in the public code repository to illustrate the data structure and input format used by the proposed method.

The complete dataset can be obtained from the corresponding author upon reasonable academic request.

⸻

Data Format

All input samples are stored as NumPy binary files (.npy) with a fixed spatial resolution of 1280 × 1280 pixels.
Each sample consists of 8 channels, organized as follows:
	1.	Channel 1: Digital Elevation Model (DEM)
	2.	Channels 2–5: Four-band remote sensing imagery
	3.	Channel 6: Lithology
	4.	Channel 7: Soil
	5.	Channel 8: Vegetation

To ensure spatial consistency, all data layers are resampled or padded to a uniform size of 1280 × 1280 before being stacked into an 8-channel array.

A representative example file (Data_instance.npy.zip) is publicly available at the following repository to demonstrate the exact data structure and channel ordering used in this study:

https://github.com/yeyeyeyeye-Liu/DSTENet.git

This example data file is intended solely for illustrating the data format and does not represent the full dataset used for model training and evaluation.

File paths and corresponding class labels are organized using a .json file, and the data are loaded through a custom Dataset class implemented in the provided code.

⸻

Public Data Sources

The raw data used to construct the dataset originate from the following publicly accessible sources:
	•	Digital Elevation Model (DEM): Shuttle Radar Topography Mission (SRTM)
	•	Remote sensing imagery: China Centre for Resources Satellite Data and Application
	•	Lithology and soil data: ISRIC – World Soil Information
	•	Vegetation data: National Cryosphere Desert Data Center

These raw datasets are preprocessed, spatially aligned, and integrated to form the final multi-channel input samples used in this study.

--------

5. License

This project is released under the MIT License. See the LICENSE file for details.

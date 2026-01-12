# DSTENet
This repository provides the implementation of DSTENet, a deep learning framework integrating Deformable RoI Pooling, ConvNeXt, and Transformer architectures for identifying hazard-prone areas of gully-type debris flows.


The code corresponds to the following paper:

A Deformable RoI Pooling and ConvNeXt–Transformer Integrated Approach for Identifying Hazard-Prone Areas of Gully-Type Debris Flows: A Case Study in Typical Regions of Yunnan, China

under review at *Computers & Geosciences*.


1. Environment

- Python 3.10  
- PyTorch 2.4.0  

The main dependencies are listed in `requirements.txt`.


2. Installation

The required Python dependencies can be installed using:

```bash
pip install -r requirements.txt


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


4. Data Availability and Format

The dataset used in this study is not publicly available due to regional data restrictions, but can be obtained from the corresponding author upon reasonable request.

Data Format
	•	Input data are stored as .npy files with a spatial size of 1280 × 1280
	•	Each sample contains 8 channels, including:
	•	Digital Elevation Model (DEM)
	•	Four-band remote sensing imagery
	•	Lithology
	•	Soil
	•	Vegetation
	•	File paths and labels are organized in a .json file
	•	Data are loaded using a custom Dataset class implemented in the code

Public Data Sources

Raw data used to construct the dataset can be obtained from the following public sources:
	•	DEM: SRTM (https://srtm.csi.cgiar.org/)
	•	Remote sensing imagery: China Centre for Resources Satellite Data and Application (https://data.cresda.cn/)
	•	Lithology and soil: ISRIC (https://data.isric.org/)
	•	Vegetation: National Cryosphere Desert Data Center (http://www.ncdc.ac.cn/)


5. License

This project is released under the MIT License. See the LICENSE file for details.

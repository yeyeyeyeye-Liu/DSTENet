# DSTENet
This repository provides the implementation of DSTENet, a deep learning framework integrating Deformable RoI Pooling, ConvNeXt, and Transformer architectures for identifying hazard-prone areas of gully-type debris flows.


The code corresponds to the following paper:

A Deformable RoI Pooling and ConvNeXt–Transformer Integrated Approach for Identifying Hazard-Prone Areas of Gully-Type Debris Flows: A Case Study in Typical Regions of Yunnan, China

currently under review at Computers & Geosciences.

--------

1. Environment

- Python 3.8  
- PyTorch 1.11 

The main dependencies are listed in `requirements.txt`.

--------

2. Installation

The required Python dependencies can be installed using:

```bash
pip install -r requirements.txt
```

--------

3. Usage

3.1 Model Training and Evaluation

The implementation of the DSTENet model, as well as the training and testing procedures, are integrated into a single script: DSTENet.py.

Training
Model training can be started by running:
```bash
python DSTENet.py --batch_size 4 --max_epoch 200 --lr 0.0005
```
Key training parameters include:
- batch_size: batch size for training (default: 4)
- max_epoch: number of training epochs (default: 200)
- lr: learning rate (default: 0.0005)
- num_workers: number of data loading workers (default: 2)
- class_num: number of output classes (default: 8)

Testing / Evaluation
The same script is used for model evaluation, depending on the configuration and checkpoint loading strategy defined in the code.

⸻

3.2 Quick Demo (Minimal Running Example)

To facilitate rapid understanding and reproducibility, a standalone demo script (demo.py) is provided.

This demo illustrates a minimal and self-contained workflow, including:
	•	Loading a representative multi-channel .npy input sample
	•	Performing necessary preprocessing and type conversion
	•	Forwarding the input through the DSTENet model
	•	Producing a valid network output without model training

The demo can be executed directly using:
```bash
python demo.py
```
The script relies on a small example dataset included in Data_instance.zip and is intended as a basic usage example to verify that the code, model architecture, and input data format are correctly configured.

--------

4. Data Availability and Format

Data Availability

The dataset used in this study is not fully publicly available due to regional data-sharing restrictions.

To facilitate data reuse and reproducibility, a representative example dataset (Data_instance.zip) is provided in the public repository to illustrate the exact input data structure, channel ordering, and file format required by the proposed method.

Researchers interested in obtaining the full dataset for academic and non-commercial research purposes may contact the corresponding author via the email address provided in the manuscript. Requests will be considered on a reasonable academic basis.

⸻

Data Format

All input samples are stored as NumPy binary files (.npy) with a fixed spatial resolution of 1280 × 1280 pixels.

Each sample consists of 8 channels, organized as follows:
- 1.	Channel 1: Digital Elevation Model (DEM)
- 2.	Channels 2–5: Four-band remote sensing imagery
- 3.	Channel 6: Lithology
- 4.	Channel 7: Soil
- 5.	Channel 8: Vegetation

To ensure spatial consistency, all data layers are resampled or padded to a uniform size of 1280 × 1280 before being stacked into an 8-channel array.

File paths and corresponding class labels are organized using a .json file.
The data are loaded through a custom PyTorch Dataset class implemented in the provided code.

The example dataset (Data_instance.zip) is intended solely to demonstrate the expected data format and does not represent the full dataset used for model training and evaluation.

Repository link:

https://github.com/yeyeyeyeye-Liu/DSTENet.git

⸻

Public Data Sources

The raw data used to construct the dataset originate from the following publicly accessible sources:
- Digital Elevation Model (DEM): Shuttle Radar Topography Mission (SRTM)
- Remote sensing imagery: China Centre for Resources Satellite Data and Application
- Lithology and soil data: ISRIC – World Soil Information
- Vegetation data: National Cryosphere Desert Data Center

These raw datasets are preprocessed, spatially aligned, and integrated to form the final multi-channel input samples used in this study.

--------

5. License

This project is released under the MIT License. See the LICENSE file for details.

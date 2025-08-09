## LinkageNet

Code written by By [Quang Huy Bui](https://scholar.google.com/citations?user=Fs_TCCsAAAAJ&hl).

This repository is an offifical implementation of the paper [End-to-End Trainable One-Stage Parking Slot Detection Integrating Global and Local Information](https://ieeexplore.ieee.org/abstract/document/9316907).


## Introduction
**Abstract.** This paper proposes an end-to-end trainable one-stage parking slot detection method for around view monitor (AVM) images. The proposed method simultaneously acquires global information (entrance, type, and occupancy of parking slot) and local information (location and orientation of junction) by using a convolutional neural network (CNN), and integrates them to detect parking slots with their properties. This method divides an AVM image into a grid and performs a CNN-based feature extraction. For each cell of the grid, the global and local information of the parking slot is obtained by applying convolution filters to the extracted feature map. Final detection results are produced by integrating the global and local information of the parking slot through non-maximum suppression (NMS). Since the proposed method obtains most of the information of the parking slot using a fully convolutional network without a region proposal stage, it is an end-to-end trainable one-stage detector. In experiments, this method was quantitatively evaluated using the public dataset (PS2.0) and outperforms previous methods by showing both recall and precision of 99.77%, type classification accuracy of 100%, and occupancy classification accuracy of 99.31% while processing 60 frames per second.




## Citation
If you find this paper useful in your research, please consider citing:
```bibtex
@ARTICLE{9316907,
  author={Suhr, Jae Kyu and Jung, Ho Gi},
  journal={IEEE Transactions on Intelligent Transportation Systems}, 
  title={End-to-End Trainable One-Stage Parking Slot Detection Integrating Global and Local Information}, 
  year={2022},
  volume={23},
  number={5},
  pages={4570-4582},
  keywords={Junctions;Feature extraction;Detectors;Task analysis;Proposals;Object detection;Deep learning;Parking slot detection;deep learning;convolutional neural network;end-to-end;one-stage detector},
  doi={10.1109/TITS.2020.3046039}}
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset preparation

- [PS2.0](https://cslinzhang.github.io/deepps/)
- [SNU](https://github.com/dohoseok/context-based-parking-slot-detect/) (for SNU dataset, download the new refined label here: [link](https://drive.google.com/file/d/1LmO-BmO7n50aWWHxdfHcZWISj8wXYNLw/view))

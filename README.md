# README 

Author: Shreyas Prasad
Class: CS 7180 - Advanced Perception
Semester: Fall 2023

Assignment 2: Color Constancy, Shadow Removal, or Intrinsic Imaging

## Description

This assignment is a implementation of the paper Color Constancy Using CNNs by Simone Bianco, Claudio Cusano, and Raimondo Schettini. The paper can be found [here](https://www.cv-foundation.org/openaccess/content_cvpr_workshops_2015/W03/papers/Bianco_Color_Constancy_Using_2015_CVPR_paper.pdf). 


## Usage

To run the program, run the following command:

```bash
python3 train.py
```


## Requirements

Run the following command to install the required packages:

```bash
pip3 install -r requirements.txt
```


## Dataset
We will use the Shi-Funt's re-processed Gehler dataset for this assignment. The dataset can be found [here](https://www.cs.sfu.ca/%7Ecolour/data/shi_gehler/). The dataset contains 568 images of 24 scenes. The images are taken under 9 different illuminations includes a variety of indoor and outdoor shots taken using two high quality DSLR cameras.

Run the following command to download the dataset:

```bash

wget www.cs.sfu.ca/~colour/data2/shi_gehler/png_canon5d_2.zip

```
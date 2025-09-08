# CoW Centerline and Feature Extraction
This repository provides a baseline algorithm for a full quantitative description of the Circle of Willis (CoW). It takes as input a multiclass segmentation mask (as provided by the TopCoW challenge) and outputs a centerline graph and surface mesh, along with derived data including the CoW variant, anatomically relevant nodes, and a rich set of morphometric features.

If you use this algorithm, please cite:
[Placeholder for arXiv citation]

We also provide a public centerline dataset built upon the TopCoW training set:
[Placeholder for Zenodo link]

<p align="center">
  <img src="https://github.com/fmusio/TopCoWSubmissions/blob/main/cnt_pipeline.png" width="500" />
</p>

## Background
The CoW is an important arterial system in the brain. Due to its central role in cerebral blood flow, it is believed to be involved in various cerebrovascular pathologies such as stroke and aneurysm. The CoW also exhibits significant inter-individual variability—both in terms of anatomical variants (e.g., hypoplastic or missing segments) and morphometric feature ranges—which further complicates its clinical assessment. This variability makes a comprehensive analysis of the CoW tedious and challenging, highlighting the need for automated tools.

### TopCoW
Voxel-level segmentation and anatomical annotation are key steps toward automated CoW analysis. To support this, we organized the challenge ["topology-aware anatomical segmentation of the Circle of Willis for CTA and MRA" (TopCoW)](https://arxiv.org/abs/2312.17670) in 2023 and 2024. The TopCoW dataset includes paired CTA and MRA scans of 200 stroke patients, with voxel-level annotations for 13 artery segments.

Both the [TopCoW training set](https://zenodo.org/records/15692630) and the [TopCoW best performing models](https://zenodo.org/records/15665435) are publicly available on Zenodo. The models can be used to segment the CoW in CTA and MRA images. Their output serves as input for our centerline and feature extraction pipeline.

### Centerline Extraction Algorithm
A full quantitative description of the CoW requires a centerline representation of the vasculature. However, extracting centerlines from segmentation masks is challenging, as conventional skeletonization algorithms often struggle with complex vascular geometries. One solution is to treat skeletonization as a segmentation task, leveraging the power of the U-Net architecture.

This repository provides an end-to-end pipeline that generates labeled centerline graphs directly from TopCoW multiclass masks. The algorithm combines the [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet) for binary skeletonization and the A* algorithm for skeleton connection. This approach enables the extraction of anatomically accurate centerline graphs and reliable morphometric features.

### Centerline Dataset
Our baseline algorithm was developed and validated using a curated centerline dataset created with the [Voreen graph generation tool](https://github.com/jqmcginnis/voreen_tools). The dataset is based on the TopCoW segmentation masks and was post-processed and verified to ensure anatomical correctness of the centerline graphs.

The training data is available on Zenodo: [Placeholder for Zenodo link]

## Prerequisites
### Install Voreen
[Voreen](https://www.uni-muenster.de/Voreen/) (Volume Rendering Engine) is a framework for the visualization and analysis of volumetric data, which can be used for [vessel graph extraction](https://github.com/jqmcginnis/voreen_tools). 

In our algorithm, the Voreen tool is used to convert voxel-based skeletons into vessel graphs, enabling a transition from the image domain to the graph domain. To use the algorithm, Voreen must be installed first.

The following instructions should work for Linux and WSL. For more information, please refer to the [build instructions](https://github.com/jqmcginnis/voreen_tools/tree/main/binaries). 

#### Install dependencies
Prepare the system for further package installations:
```
sudo apt-get update && sudo apt-get install -y --no-install-recommends apt-utils
```
And install the dependencies:
```
sudo apt install -y g++ cmake libboost-all-dev libglew-dev libqt5svg5-dev libdevil-dev ffmpeg libswscale-dev libavcodec-dev libavformat-dev build-essential mesa-common-dev mesa-utils freeglut3-dev qtbase5-dev qt5-qmake qtbase5-dev-tools ninja-build libhdf5-dev liblz4-dev python3-dev
```

#### Build VTK
You need to build VTK as a prerequisite for the Voreen tool.   
**NOTE:** You might want to change directories (~/projects) in the commands below!
``` wget https://www.vtk.org/files/release/9.2/VTK-9.2.6.tar.gz
mkdir ~/projects
tar -xf VTK-9.2.6.tar.gz -C  ~/projects
mv  ~/projects/VTK-9.2.6  ~/projects/vtk_source
mkdir ~/projects/vtk
cd  ~/projects/vtk_source && cmake . -B  ~/projects/vtk
cd  ~/projects/vtk && make -j 8
```

#### Build Voreen
Build the Voreen tool with certain settings enabled in the cmake file.   
**NOTE:** You might want to change directories (~/projects) in the commands below!
```
wget https://github.com/jqmcginnis/voreen_tools/raw/main/binaries/voreen-src-unix-nightly.tar.gz
tar -xf voreen-src-unix-nightly.tar.gz -C ~/projects
printf 'set(VRN_MODULE_BASE ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_BIGDATAIMAGEPROCESSING ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_CONNEXE ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_DEPRECATED OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_DEVIL ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_ENSEMBLEANALYSIS ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_EXPERIMENTAL OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_FFMPEG OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_FLOWANALYSIS ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_GDCM OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_HDF5 ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_ITK OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_ITK_GENERATED OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_OPENCL OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_OPENMP OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_PLOTTING ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_POI OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_PVM ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_PYTHON ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_RANDOMWALKER ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_SAMPLE OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_SEGY ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_STAGING ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_STEREOSCOPY ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_SURFACE ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_TIFF OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_ULTRAMICROSCOPYDEPLOYMENT OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_VESSELNETWORKANALYSIS ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_VOLUMELABELING OFF CACHE BOOL "" FORCE)\nset(VRN_MODULE_VTK ON CACHE BOOL "" FORCE)\nset(VRN_MODULE_ZIP ON CACHE BOOL "" FORCE)\nset(VRN_NON_INTERACTIVE OFF CACHE BOOL "" FORCE)\nset(VRN_OPENGL_COMPATIBILITY_PROFILE OFF CACHE BOOL "" FORCE)\nset(VRN_PRECOMPILED_HEADER OFF CACHE BOOL "" FORCE)\nset(VRN_USE_GENERIC_FILE_WATCHER OFF CACHE BOOL "" FORCE)\nset(VRN_USE_HDF5_VERSION 1.10 CACHE STRING "" FORCE)\nset(VRN_USE_SSE41 ON CACHE BOOL "" FORCE)\nset(VRN_VESSELNETWORKANALYSIS_BUILD ON CACHE BOOL "" FORCE)\nset(VRN_BUILD_VOREENTOOL ON CACHE BOOL "" FORCE)\nset(VTK_DIR ~/projects/vtk/lib/cmake/vtk-9.2 CACHE PATH "" FORCE)' >> ~/projects/voreen-src-unix-nightly/config-default.cmake
cd ~/projects/voreen-src-unix-nightly
mkdir build && cd build
cmake ..
make -j 8
```

### Create Python Environment
Once Voreen is installed properly, you  need to set up a python environment for the remaining dependencies. We strongly recommend that you create a separate virtual environment to run the pipeline.

#### Install nnUNet
The following is from the [nnUNet installation instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/installation_instructions.md):

1) Install [PyTorch](https://pytorch.org/get-started/locally/) as described on their website (conda/pip). Please 
install the latest version with support for your hardware (cuda, mps, cpu).
**DO NOT JUST `pip install nnunetv2` WITHOUT PROPERLY INSTALLING PYTORCH FIRST**.  
2) Install nnU-Net using the provided copy of the nnU-Net codebase:
```bash
cd predict_skeleton/nnUNet
pip install -e .
```
#### Install remaining dependencies
Run the following command from the project root:
```bash
pip install -r requirements.txt
```

## Usage
The pipeline runs on TopCoW multiclass masks, which can be produced by running one of the [publicly available models](https://zenodo.org/records/15665435) on CTA and/or MRA data. 

**Pipeline Input**
- Multiclass segmentation mask (.nii.gz): A mask with 13 labeled artery segments, as defined by the TopCoW challenge.

**Pipeline steps**
1) **Process mask:** Crop to foreground and resample to a common spacing. Optionally, apply corrections at the mask level.
2) **Predict skeleton:** Run nnUNet inference on the mask to produce a binary skeleton.
3) **Connect skeleton:** Label and connect the skeleton using the A* algorithm.
4) **Extract Voreen graph:** Convert the skeleton into a vessel graph using Voreen.
5) **Postprocessing graph:** Apply postprocessing steps including variant and node extraction, as well as smoothing.
6) **Compute radius:** Perform cross-sectional analysis along the centerline to compute MIS and CE radius for each edge.
7) **Extract features:** Compute a set of segment and bifurcation features, including fetal PCA status.

**Pipeline Output**
- Centerline graph (.vtp): Labeled centerline representation of the CoW
- Surface mesh (.vtp): Labeled surface mesh of the segmented vasculature
- Variant file (.json): Encodes the anatomical CoW variant
- Node file (.json): Contains information on anatomically relevant nodes (e.g. bifurcations, segment boundaries)
- Feature file (.json): Includes morphometric features for segments and bifurcations

### Configure pipeline
The pipeline can be configured using the configs.py file.

You have to specify the following paths:
- *cow_mlt_seg_dir*: path to the directory containing your input masks 
- *voreen_tool_path*: path to voreen tool binaries
- *nnunet_results*: path to the nnUNet model weights (see below)

Additionally, you must specify the image modality ('ct' or 'mr') and a set of pipeline parameters. Default parameters are provided, so specifying the paths and modality is sufficient to get started.

### Model weights
The model weights for skeletonization are available for download on Zenodo: [Insert Zenodo link here]

Download the weights and place them in the folder defined by *nnunet_results*, using the same relative path structure as specified on Zenodo.

### Run the pipeline
Provide at least one multiclass mask in the folder specified by *cow_mlt_seg_dir*, and ensure the model weights are correctly placed. Once everything is set up, simply run the main script without any arguments:
```bash
python3 main.py
```

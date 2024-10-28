# Behavioral_Tagging_of_Mice_in_multiple_Mice_dataset_using_Deep_Learning
*Identification and temporal classification of social behavior of different mice with different pathologies in order to study them.*

## Overview
The study of social behavior in mice is a crucial aspect of neuroscience research, particularly in understanding the effects of various pathologies or treatments. One of the key methods used in this research is the open-field test, where the behavior and social interactions of mice are observed in an open, stimulus-free environment. Traditionally, analyzing these tests is a manual, time-consuming process, with increasing complexity as the number of mice and interactions grows. Additionally, subjective interpretations can introduce bias, leading to inconsistencies in the results.

This repository addresses these challenges by providing an automated solution for analyzing open-field test videos. Using a top-view configuration, the system leverages **DeepLabCut (DLC)** for pose estimation to track key body parts of multiple animals over time and leverages **Graph Attention Networks (GATs)** for behavior recognition. The end-to-end pipeline is designed to extract poses from videos, construct spatio-temporal graphs, and classify various social behaviors between mice.

## Project Structure:
```
src
├── augmentation.py
├── baseline_models/
├── baseline_models.ipynb
├── DataDLC.py
├── dataloader.py
├── gui.py
├── models.py
├── preprocessing.py
├── results_baseline_models.ipynb
├── train.py
├── train_poursuit.py
├── utils.py
├── utils_deepof.py
├── Visualization.ipynb
```

## Methodology

### 1. Pose Estimation

The first part of the pipeline involves using **DeepLabCut (DLC)** for extracting poses of multiple animals from videos. Each video is processed to obtain coordinates of key body parts (e.g., nose, ears, tail) for every frame, generating a time-series of pose data.

**Methods:**
- **DeepLabCut Model Fine-Tuning:** A fine-tuned ResNet model is used for pose estimation in the specific multi-animal setup.
- **Pose Data Preprocessing:** Raw pose data is cleaned and normalized to remove noise and standardize keypoint positions across frames.

### 2. Spatio-Temporal Graph Construction 

Pose data is transformed into a **spatio-temporal graph representation**, where nodes represent body parts, and edges represent both spatial relationships (connections between body parts) and temporal relationships (connections between the same body part across frames).

**Methods:**
- **Graph Construction:** Each frame is represented as a graph, with nodes for each body part and edges capturing spatial proximity. Temporal edges link the same nodes across sequential frames, forming a **graph sequence**.
- **Graph Data Preparation:** Graphs are packaged as `torch_geometric` `Data` objects, including node features (e.g., keypoint coordinates, individual identity) and labels for the behaviors present in each frame sequence.

### 3. Behavior Recognition using Graph Attention Networks (GATs)

The core of behavior recognition is a Graph Attention Network (GAT) encoder followed by a Classification Head based on fully connected layers. A separate GAT-based model is trained for each behavior to focus on behavior-specific features.

**Methods:**
 - **Graph Attention Network (GAT):** Extracts a spatio temporal features from the spatio-temporal graph.
- **Multi-Class Classification:** Each model is trained to recognize the presence or absence of a specific behavior (e.g., Sniffing, Dominance).

## Getting Started
### Prerequisites
- Python >=3.8
- Instal dependencies using:
```
pip install -r requirements.txt
```
- **DeepLabCut** for pose estimation.
- **PyTorch Geometric** for the spatio-temporal graph.

#### Setup
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/multi-animal-behavior-recognition.git
   cd multi-animal-behavior-recognition
   ```
2. Download pre-trained models and place them in the `models/` directory.
3. Prepare pose estimation data and place it in the `data/` directory.

#### Running the GUI
Launch the GUI for data loading and result analysis:
```
python gui/main_gui.py
```






  ---
  
- src/dataloader.py: Contains the class Data_DLC, which loads the output file '.h5' of DeepLabCut into a pandas MultiIndex data frame and allows manipulation and pre-processing ot the time-series. Documentation on the specifically functionalities can be found in ...
- src/Models.py: Contains the class that defines the models to be used.
- 
## Description of Files

### `DataDLC.py`
`DataDLC.py` is a core module for loading, preprocessing, and cleaning data from pose estimation .h5 files. This class is designed to handle the raw coordinates of tracked body parts and prepares the data for further analysis, including building graphs for pose-based analysis. Here’s an overview of the class and its main methods:

Class: DataDLC
Handles loading and preprocessing data from .h5 files generated from pose estimation models like DeepLabCut.
Stores key attributes such as the number of individuals, body parts, and frames.
Processes and cleans coordinates to ensure data consistency and prepares it for downstream tasks.
The DataDLC class also includes various methods for data imputation, statistical analysis, and visualization. These functionalities enable handling missing data, creating videos for better visualization of the tracked points, and saving the processed data.
Main Methods:

- **`__init__(self, file: str, detect_jumps: bool = False)`**: Initializes the DataDLC class by loading data from the specified .h5 file and optionally detecting and correcting isolated jumps.

- **`load_data(self, detect_jumps: bool)`**: Loads data from the provided file, extracts coordinates for each individual, computes the center of mass, and handles NaN values and isolated jumps.

- **`compute_center_of_mass(self)`**: Calculates the center of mass for each individual, excluding tail points to focus on core body parts.

- **`clean_inconsistent_nans(self)`**: Ensures that if either the x or y coordinate of a body part is NaN, the entire coordinate set for that frame is set to NaN.

- **`fill_nans(self)`**: Fills NaN values with zeros for consistency in further analysis.

- **`detect_isolated_jumps(self, threshold_soft_min: int, threshold_soft_max: int, imputation: bool)`**: Detects isolated jumps in time-series data using thresholds for minimal and maximal jumps, and optionally imputes these jumps using linear interpolation.

- **`remove_outlier_tracklets(self, threshold_split_tracklets: int, threshold_jump: int, percentage_gap_neigh: float, verbose: bool)`**: Identifies and removes outlier tracklets that deviate significantly from neighboring tracklets.

- **`detect_tracklets(self, x, y, threshold: int)`**: Detects tracklets (continuous segments of valid data points) in the time-series data based on specified thresholds.

- **`entropy_of_masks(self, mask1, mask2)`**:Calculates the entropy between two masks, which can be used to compare differences between two datasets.

- **`drop_tail_bodyparts(self)`**:Removes body parts corresponding to the tail from the dataset before building the graph, reducing the data dimensions for specific analyses.

- **`create_video(self, video_path, output_path, plot_prev_coords=False, frames=None)`**:Creates a video showing the tracked body parts of each individual over time.
Includes options to plot previous coordinates and specify a range of frames to process.

- **`get_statistics_on_jumps(self, plot=False)`**:Computes the mean and standard deviation of jumps between consecutive points, assuming a Gaussian distribution for jumps.
If `plot=True`, displays a histogram of the jumps for better visualization.

- **`create_video_per_event(self, video_path, output_path, events, split_behaviour=False)`**:Generates videos highlighting specific events on each frame.
If split_behaviour=True, creates separate videos for each event; otherwise, it overlays all events on a single video.
Useful for visualizing behavior annotations alongside the tracked points.

- **`save(self, path)`**:Saves the processed data back into an `.h5` file, preserving the changes made during analysis.
Ensures compatibility with other tools by storing the data in a structured format.


This class provides a robust way to prepare time-series pose estimation data for further analysis, addressing common issues such as jumps, outliers, and inconsistent coordinates.

### `augmentation.py`

This script contains functions for augmenting and balancing datasets of mouse behavior data, specifically targeting symmetrical behaviors and class imbalances. The functions included are:

- **`merge_symetric_behaviours()`**: Merges two symmetrical behaviors by swapping the identities of the subjects and combining occurrences of the two behaviors into one. This is useful for combining behaviors like 'Sniffing_Resident' and 'Sniffing_Visitor' into a single category while maintaining identity distinctions in the dataset.

- **`rotate_samples()`**: Rotates samples in the dataset based on active behaviors. The function creates symmetry by flipping the pose data along the x or y axis, transposing the coordinates, or rotating them by 180 degrees. This augmentation helps the model generalize better by providing additional variations of the behaviors.

- **`downsample_inactive()`**: Balances the dataset by randomly selecting a subset of inactive samples to match the number of active samples for a specific behavior. This helps in reducing class imbalance, especially when the inactive instances are significantly higher in number.

- **`downsample_majority_class()`**: Downsamples the majority class (either active or inactive samples) to match the count of the minority class for a specified behavior. It aims to maintain class balance, reducing the risk of the model being biased towards the more frequent class.

- **`merge_symetric_behaviours_version2()`**: Similar to `merge_symetric_behaviours()`, but it creates new samples for all instances of a behavior in the secondary individual, preserving additional context. This function is designed to help the model differentiate between individuals while keeping both behaviors represented.

- **`merge_symetric_behaviours_sequences()`**: Applies the merging of symmetrical behaviors on sequences of data, adjusting the identity labels across multiple frames. This is useful for scenarios where the dataset contains time-series data, allowing consistent merging of behaviors across frames while maintaining individual identities.

These functions support data augmentation, balancing, and preparation for training machine learning models on behavior recognition tasks in mice.







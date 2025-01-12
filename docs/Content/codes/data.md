# LHAI Project Code: `data` Folder  

The `data` folder is designated for storing datasets required for training, inference, and evaluation. It is organized by data types to ensure easy access and effective version management.  

## Folder Structure  

```plaintext
├── data               <- Different types of datasets (recommended in `.npy` format)
    ├── FERMI          <- Datasets related to Fermi observations or simulations
    ├── POISSON        <- Poisson-distributed datasets for statistical modeling
    ├── SIMU           <- Synthetic datasets for simulation purposes
    └── RAW            <- Unprocessed raw data obtained from experiments or external sources
```

## Folder Details  

### FERMI  
- **Description**: Contains observational data or related simulation datasets from the Fermi Gamma-ray Space Telescope.  
- **Usage**: For preprocessing workflows or training models in gamma-ray studies.  
- **Examples**:  
  - `FERMI_2024_observations.npy`: Preprocessed data from the 2024 observation period.  
  - `FERMI_training_split.npy`: Subset of data prepared for model input during training.  

### POISSON  
- **Description**: Stores datasets based on Poisson distributions, commonly used for event simulations or error modeling.  
- **Usage**: Statistical modeling, event rate prediction, or uncertainty quantification.  
- **Examples**:  
  - `Poisson_noise_samples.npy`: Synthetic samples with Poisson noise.  
  - `Poisson_event_counts.npy`: Event count distribution for analysis.  

### SIMU  
- **Description**: Contains synthetic datasets created for specific scenarios or experimental simulations.  
- **Usage**: Model validation, sensitivity analysis, or testing new algorithms.  
- **Examples**:  
  - `Simu_100k_events.npy`: A simulated dataset with 100,000 event records.  
  - `Simu_input_parameters.json`: Configuration file used to generate synthetic data.  

### RAW  
- **Description**: This folder contains unprocessed raw datasets directly obtained from instruments or collaborators.  
- **Usage**: Starting point for preprocessing workflows or for manual analysis.  
- **Examples**:  
  - `Raw_experiment_2024.dat`: Data dump from the latest experiment.  
  - `Raw_sensor_readings.bin`: Binary sensor readings requiring decoding.  

## File Format Requirements for Training and Inference  

This section is crucial and the file storage guidelines are under development.

<p align='right'>by Zihang Liu</p>
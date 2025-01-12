# LHAI Project Code: `data` Folder

The `data` folder is the repository for datasets required for training, inference, and evaluation. It is structured to organize data by type and facilitate streamlined access and versioning.  

## Folder Structure  

```plaintext
├── data               <- Different types of data. (Recommended to upload in `.npy` format)
    ├── FERMI          <- Datasets related to Fermi observations or simulations.
    ├── POISSON        <- Poisson-distributed datasets for statistical analysis or modeling.
    ├── SIMU           <- Synthetic data generated for simulation purposes.
    └── RAW            <- Unprocessed raw data files as received from experiments or external sources.
```

## Folder Details

### FERMI
- **Description**: Contains data from the Fermi Gamma-ray Space Telescope or related simulation datasets.  
- **File Formats**: Commonly in `.npy`, `.csv`, or `.fits` formats.  
- **Usage**: Used in preprocessing pipelines for gamma-ray studies or training models.  
- **Examples**:  
  - `FERMI_2024_observations.npy`: Preprocessed data from the 2024 observation period.  
  - `FERMI_training_split.npy`: Training subset for model input.

### POISSON  
- **Description**: Houses datasets based on Poisson distributions, often used in event-based simulations or error modeling.  
- **File Formats**: `.npy`, `.txt`.  
- **Usage**: Statistical models, event rate predictions, or uncertainty quantification.  
- **Examples**:  
  - `Poisson_noise_samples.npy`: Synthetic samples with Poisson noise.  
  - `Poisson_event_counts.npy`: Event count distributions for analysis.  

### SIMU  
- **Description**: Stores simulated datasets that replicate specific scenarios or experiments.  
- **File Formats**: `.npy`, `.mat`, `.json`.  
- **Usage**: Model validation, sensitivity analysis, or testing new algorithms.  
- **Examples**:  
  - `Simu_100k_events.npy`: Simulated dataset with 100,000 event records.  
  - `Simu_input_parameters.json`: Configuration file for generating synthetic data.

### RAW  
- **Description**: This folder contains unprocessed raw datasets received directly from instruments or collaborators.  
- **File Formats**: `.txt`, `.bin`, `.dat`.  
- **Usage**: Starting point for preprocessing pipelines or manual analysis.  
- **Examples**:  
  - `Raw_experiment_2024.dat`: Data dump from the latest experiment.  
  - `Raw_sensor_readings.bin`: Binary file from sensors, requiring decoding.  

## Best Practices  

1. **File Naming**: Use clear, consistent names that indicate dataset type, source, and version.  
   - Example: `FERMI_2023_v1.npy`, `SIMU_test_run_001.npy`.  

2. **File Formats**: Prefer `.npy` format for compatibility and efficient loading with Python tools like NumPy.  

3. **Documentation**: Include a metadata file (e.g., `README.md`) in each subfolder to document dataset origins, formats, and preprocessing steps.  

4. **Data Integrity**: Use checksums or hashes to verify data integrity when sharing or archiving files.  

5. **Versioning**: If datasets are updated, maintain version control to track changes and ensure reproducibility.  

6. **Storage Limits**: For large datasets, consider using cloud storage or external repositories and linking them to the project.  
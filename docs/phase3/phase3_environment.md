# GeoSense Phase III - Environment and Reproducibility



## 1. Purpose



This document records the verified Python environment and direct dependencies used by the GeoSense Phase III LiDAR and street-view analysis modules. It provides setup instructions and describes limitations that may affect reproducibility on another computer.



## 2. Verified Environment



The following details were verified on 12 October 2026.



| Component | Verified configuration |

|---|---|

| Operating system | Windows |

| Python | 3.11.17 |

| PyTorch | 2.5.1+cu121 |

| CUDA available | True |

| NumPy | 2.4.6 |

| GeoPandas | 1.1.3 |

| Rasterio | 1.4.4 |



Python executable:



`D:\GeoSense_Phase3\geosense_lidar\python.exe`



PyTorch detected CUDA support in the verified environment. Actual GPU execution should be tested after installation on another computer.



## 3. Phase III Dependencies



The file `requirements-phase3.txt` records these direct dependencies:



- `laspy` - LiDAR point-cloud processing.

- `numpy` - numerical and array operations.

- `pandas` - tabular data processing.

- `pyproj` - coordinate transformations.

- `requests` - HTTP requests to external services.

- `python-dotenv` - loading environment variables.

- `Pillow` - image loading and handling.

- `segment-anything` - Segment Anything Model (SAM) inference.

- `transformers` - transformer-based semantic segmentation using SegFormer.



PyTorch is also required for SAM and SegFormer. The verified version is `2.5.1+cu121`. Install a compatible PyTorch build separately because the correct build depends on the target computer and its GPU configuration.



## 4. Setup and Verification



To recreate the environment:



1. Install Python 3.11.

2. Create and activate a dedicated Python environment.

3. Install a compatible PyTorch build using the official PyTorch installation instructions.

4. Install the dependencies listed in `requirements-phase3.txt`.

5. Verify the installed packages and CUDA availability.

6. Run the Phase III automated tests.



Install the listed dependencies with:



`python -m pip install -r requirements-phase3.txt`



Verify the main imports and PyTorch configuration with:



`python -c "import sys, torch, laspy, numpy, pandas, pyproj, requests, dotenv, PIL, segment_anything, transformers; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"`



Run the Phase III test suite with:



`D:\GeoSense_Phase3\geosense_lidar\python.exe -m pytest D:\GeoSense_Phase3\tests\test_phase3_modules.py -v`



A previous test run passed 11 tests. Run the tests again after recreating the environment to verify the new setup.



## 5. Configuration, Models, and Data



### Mapillary configuration



The Mapillary downloader reads `MAPILLARY_ACCESS_TOKEN` from a local `.env` file. Supply the token locally. Never commit API credentials or other secrets to GitHub.



### SAM model checkpoint



The verified SAM checkpoint is located at:



`D:\GeoSense_Phase3\models\saved\sam\sam_vit_b_01ec64.pth`



The checkpoint must be available at the configured path, or the code must be updated to use the appropriate path on the target computer.



### Project paths



Some Phase III modules use absolute paths under `D:\GeoSense_Phase3`. These paths are specific to the current computer and may need to be updated before the project can run elsewhere.



### Data requirements



The dependency file does not include LiDAR datasets, downloaded street-view imagery, API credentials, or model checkpoint files. These resources must be obtained and configured separately.



## 6. Reproducibility Limitations



- The dependency file lists direct dependencies but does not lock every transitive dependency.

- CUDA-enabled PyTorch installation requirements may differ between computers.

- Model checkpoints and input datasets are not included in the dependency file.

- Absolute Windows paths may require modification.

- Passing automated tests does not independently validate every geospatial measurement or semantic-segmentation result.

- The LiDAR and Chennai street-view datasets represent different geographic areas. Their outputs should not be interpreted as measurements from the same site without appropriate geographically matched data.



## 7. Maintenance and Status



Update this document when the verified Python version, dependency versions, model checkpoints, configuration paths, or test results change.



Related dependency file: `requirements-phase3.txt`



Status: Environment details documented. Reproduction on another computer and full workflow validation remain to be tested.

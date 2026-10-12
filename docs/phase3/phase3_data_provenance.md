\# GeoSense Phase III - Data Provenance and Licensing Guide



\## 1. Purpose



This document establishes a record-keeping framework for the data, imagery, software libraries, and model checkpoints used in GeoSense Phase III. It supports research reproducibility, responsible data use, attribution, and transparent reporting of data limitations.



This guide distinguishes verified project configuration from information that must be confirmed using the original dataset or provider documentation.



\## 2. Data Provenance Requirements



For every dataset used in the project, maintain a record containing:



\- Dataset name and provider.

\- Original download or access URL.

\- Date accessed or downloaded.

\- Acquisition date or observation period, where available.

\- Geographic coverage and spatial resolution.

\- Coordinate reference system.

\- File format and processing history.

\- Licence, attribution requirements, and permitted uses.

\- Any preprocessing, clipping, filtering, or conversion applied.



Do not assume that a dataset is openly licensed merely because it can be downloaded or accessed through an API.



\## 3. LiDAR Point-Cloud Data



The Phase III analysis uses LiDAR point-cloud data for point classification, terrain-related analysis, and derived site characteristics.



\*\*Information to document:\*\*

\- Original data provider and dataset identifier.

\- Download URL and acquisition date.

\- Survey or collection date, if available.

\- Geographic coverage and coordinate reference system.

\- Point-cloud format, such as LAS or LAZ.

\- Point density and classification attributes, where documented.

\- Licence and attribution requirements.

\- Any cropping, filtering, reprojection, or derived calculations.



\*\*Known project limitation:\*\* The previously analysed LiDAR dataset did not cover Chennai. The absence of points within the Chennai study area must not be interpreted as evidence that buildings, vegetation, or other surface features are absent.



The original provider, licence, acquisition date, and complete spatial coverage must be verified from the source dataset and its accompanying metadata before publication.



\## 4. Street-View and Mapillary Imagery



Street-view imagery may be used for visual interpretation and exploratory semantic analysis.



Record:

\- Imagery provider and API endpoint.

\- Image identifier or other permitted reference.

\- Retrieval date and image capture date, if available.

\- Image location and spatial coverage.

\- API access conditions and applicable licence.

\- Attribution and redistribution requirements.

\- Image preprocessing and any derived outputs.



The project uses the environment variable `MAPILLARY\_ACCESS\_TOKEN` for Mapillary authentication.



\*\*Security requirements:\*\*

\- Store access tokens locally in an appropriately protected `.env` file.

\- Do not commit `.env` files, tokens, or authentication headers to Git.

\- Do not include secrets in logs, screenshots, or public documentation.

\- Review the provider's current terms before redistributing imagery or derived products.



An API-access token does not automatically grant unrestricted rights to reuse or redistribute imagery.



\## 5. Satellite and Ancillary Data



If satellite imagery, digital elevation models, land-cover products, or other ancillary datasets are incorporated into Phase III, document each source separately.



For each source, record:

\- Satellite, sensor, product, or dataset name.

\- Provider and official product documentation.

\- Acquisition dates and processing level.

\- Spatial and temporal resolution.

\- Coordinate reference system and pixel size.

\- Cloud filtering, mosaicking, clipping, or other preprocessing.

\- Licence, attribution, and publication requirements.



Do not list a satellite dataset as an input to a particular analysis unless its use has been confirmed in the corresponding workflow.



\## 6. Model Checkpoints and Pretrained Models



The project uses the Segment Anything Model (SAM) checkpoint at the configured path:



`D:\\GeoSense\_Phase3\\models\\saved\\sam\\sam\_vit\_b\_01ec64.pth`



For each pretrained model, record:

\- Model name, architecture, and variant.

\- Original model repository and download URL.

\- Model version or release identifier.

\- Checkpoint filename and, where feasible, a SHA-256 checksum.

\- Model licence and attribution requirements.

\- Intended use and known limitations.

\- Any fine-tuning, prompt configuration, or parameter changes.



Model weights may have licensing conditions that differ from the licence of the surrounding Python library. Verify the applicable terms before redistributing checkpoints or using outputs in restricted contexts.



\## 7. Software and Python Dependencies



The repository contains a Phase III dependency specification in `requirements-phase3.txt`.



The verified environment previously reported:

\- Python 3.11.17.

\- PyTorch 2.5.1+cu121.

\- CUDA availability: True.

\- NumPy 2.4.6.

\- GeoPandas 1.1.3.

\- Rasterio 1.4.4.



These values describe the previously verified environment and may differ on another computer.



For reproducibility:

\- Record the operating system, Python version, and package versions.

\- Preserve the dependency specification used for the analysis.

\- Document the PyTorch and CUDA configuration separately.

\- Review each dependency's licence when preparing a distributable software package.

\- Do not assume that the project's dependency list captures every installed package or external system requirement.



\## 8. Processing History and Derived Outputs



Derived datasets and maps should be traceable to their original inputs.



Record:

\- Input filenames and source identifiers.

\- Processing script or notebook.

\- Processing date and software versions.

\- Coordinate transformations and spatial filters.

\- Parameters, thresholds, model settings, and random seeds where relevant.

\- Output filename, format, resolution, and coordinate reference system.

\- Known errors, exclusions, or manual interventions.



For maps and research figures, include suitable data-source acknowledgements and distinguish observed data from model-derived estimates.



\## 9. Attribution and Redistribution



Before publishing or sharing any dataset, imagery, model checkpoint, map, or derived output:



1\. Identify the original provider.

2\. Read the applicable licence and current terms of use.

3\. Confirm whether commercial use, modification, and redistribution are permitted.

4\. Include required citations and acknowledgements.

5\. Respect restrictions on personal information, location information, and imagery redistribution.

6\. Record the date on which the licence or terms were checked.



Where licence information has not been verified, mark it as \*\*Pending verification\*\* rather than assuming that redistribution is permitted.



\## 10. Research Limitations and Interpretation



Data provenance does not itself establish scientific validity. The following limitations should be considered when reporting Phase III results:



\- LiDAR-derived point percentages are proxies and should not automatically be interpreted as land-area percentages.

\- Derived slope-risk scores must be explained using the implemented calculation and its limitations.

\- Street-view semantic classifications require manual validation before being treated as reliable measurements.

\- Rooftop segmentation outputs depend on model configuration, prompts or settings, and image conditions.

\- LiDAR and Chennai street-view imagery represent different locations and should not be combined as observations from a common site.

\- Successful software tests do not independently validate every derived metric or research conclusion.



\## 11. Recommended Provenance Register



Maintain a separate register with one entry per dataset or model.



| Field | Description |

|---|---|

| Asset ID | Unique identifier |

| Asset name | Dataset, image collection, or checkpoint |

| Provider | Original source |

| Source URL | Official source or download page |

| Access date | Date obtained |

| Observation date | Date collected, if known |

| Coverage | Geographic extent |

| Licence | Verified licence or terms |

| Attribution | Required acknowledgement |

| Processing | Transformations and scripts |

| Status | Verified or Pending verification |



\## 12. Maintenance and Status



Update this guide whenever a new data source, model checkpoint, or dependency is introduced. Preserve the original source metadata and processing history alongside the project outputs.



This document is a provenance framework, not a claim that every data source or licence has already been verified. Complete the pending source and licence records before final dissertation submission or public redistribution.


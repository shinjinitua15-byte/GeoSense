\# GeoSense Agent 2.0 — Phase III Practicum Notes



\## 1. Discussion: Satellite Imagery, LiDAR and Street-Level Imagery



Satellite imagery provides valuable information about land cover, vegetation, built-up areas and the spatial distribution of surface features. However, conventional satellite imagery alone may not provide sufficient detail about the height and three-dimensional structure of individual objects or pedestrian-level conditions, such as sidewalk quality and street furniture.



LiDAR (Light Detection and Ranging) uses laser pulses to obtain three-dimensional measurements of objects and terrain. In GeoSense Phase III, LiDAR data are used to estimate building heights, canopy point share, impervious point share and a slope-risk indicator. Street-level imagery provides a ground-level perspective that can help identify roads, sidewalks, vegetation and other visible streetscape features.



For example, when selecting two plots for a new clinic, satellite imagery could help compare surrounding land cover, accessibility patterns and built-up development. LiDAR could provide additional information about building heights and terrain structure, while street-level imagery could help assess visible sidewalk conditions and the surrounding streetscape. Combining these sources can provide a more detailed understanding than relying on satellite imagery alone.



However, these datasets must cover the same study area and period wherever possible. The GeoSense example LiDAR dataset covers a different geographic region from the Chennai Centre street-view example. Therefore, their results cannot be interpreted as measurements of one common location.



\## 2. Comparison of Phase I, Phase II and Phase III



| Aspect | Phase I | Phase II | Phase III |

|---|---|---|---|

| Main focus | Site-suitability analysis | Deep-learning-based imagery analysis | LiDAR and street-view analysis |

| Data perspective | Depends on the available Phase I inputs | Overhead imagery and model inputs | Three-dimensional point clouds and street-level imagery |

| Main capability | Provides suitability indicators or predictions when available | Supports image classification or segmentation when available | Extracts point-cloud metrics and streetscape indicators |

| Example outputs | Suitability results | Imagery-classification results | LiDAR metrics, semantic-analysis results and JSON site profiles |

| Contribution to a combined assessment | Supplies suitability information | Supplies imagery-derived information | Adds structural and ground-level context |



\*\*Integration status:\*\* The Phase III modules and profile-generation workflow have been demonstrated, and the Phase III test suite passed 11 tests. However, Phase I suitability predictions and Phase II imagery-classification outputs are not yet available in the generated unified profiles. Complete end-to-end integration remains pending.



\## 3. Major Phase III Outputs



\### 3.1 LiDAR Analyzer — `lidar\_analyzer.py`



\*\*Input:\*\* LiDAR point-cloud data and a geographic bounding box.



\*\*Processing:\*\* Filters points within the requested area and calculates point-cloud-based metrics.



\*\*Output:\*\* Building-height estimates, canopy point share, impervious point share and a slope-risk indicator.



\*\*Importance for Phase IV:\*\* These metrics can contribute structural and terrain information to a broader site-assessment workflow. Their accuracy and interpretation should be validated against independent reference data before operational use.



The available LiDAR study area contains approximately 1.12 million points. The reported impervious metric is a point-share proxy and should not be interpreted as a validated estimate of total impervious land area.



\### 3.2 Street-View Semantic Analysis — `streetview\_semantic.py`



\*\*Input:\*\* Available street-level images.



\*\*Processing:\*\* Analyzes visible streetscape components, including roads, sidewalks and vegetation.



\*\*Output:\*\* Image-derived semantic proportions and related streetscape indicators.



\*\*Importance for Phase IV:\*\* Street-level information can complement overhead imagery by representing features visible from a pedestrian perspective.



The semantic results are exploratory. Classification accuracy, image coverage and the reliability of derived walkability indicators require further validation.



\### 3.3 Unified Enhanced Site Profile — `unified\_site\_profile.py`



\*\*Input:\*\* Geographic coordinates, a search radius and the available Phase III analysis results.



\*\*Processing:\*\* Organizes available information into a structured site profile.



\*\*Output:\*\* A JSON report containing available LiDAR, street-view and other phase-related fields.



\*\*Importance for Phase IV:\*\* A standardized profile provides a common structure for combining outputs from multiple analysis modules. The integration should only be considered complete when actual Phase I and Phase II outputs are connected and verified.



The Chennai Centre example profile reports zero matching LiDAR points because the available point cloud does not cover Chennai. Missing Phase I and Phase II outputs remain unavailable rather than being treated as successful predictions.



\## 4. Troubleshooting and Corrective Actions



\### 4.1 MemoryError During Point-Cloud Processing



\*\*Cause:\*\* Large point clouds can require more memory than is available when loading or processing all points simultaneously.



\*\*Corrective actions:\*\*

\- Process a smaller geographic bounding box.

\- Downsample the point cloud when appropriate.

\- Process the data in manageable chunks.

\- Check memory use and the size of intermediate arrays.

\- Preserve the original dataset before applying destructive changes.



\### 4.2 No CUDA GPUs Available



\*\*Cause:\*\* PyTorch may not detect a compatible GPU because of an incompatible installation, driver configuration or execution environment.



\*\*Corrective actions:\*\*

\- Check whether `torch.cuda.is\_available()` returns `True`.

\- Verify the installed PyTorch build and compatible GPU drivers.

\- Use CPU execution if GPU acceleration is unavailable.

\- Consider a compatible GPU-enabled environment when model processing requires acceleration.



GPU availability alone does not guarantee that every model or operation will run successfully.



\### 4.3 Empty Mapillary Results



\*\*Possible causes:\*\* The selected area may have limited street-view coverage, the geographic bounding box may be incorrect, authentication may be invalid, or the service may impose request limits.



\*\*Corrective actions:\*\*

\- Verify the coordinates and bounding box.

\- Check image coverage for the intended location.

\- Confirm that authentication is configured correctly without exposing the access token.

\- Inspect the service response and any rate-limit or permission errors.

\- Record zero-result cases instead of inventing missing imagery.



\### 4.4 PDAL Command or Import Not Found



\*\*Cause:\*\* PDAL may not be installed, may not be available on the system path, or may not be accessible from the active Python environment.



\*\*Corrective actions:\*\*

\- Activate the intended environment.

\- Check whether the PDAL command or Python package is available.

\- Install a compatible version using an appropriate package source if required.

\- Verify the installation before rerunning the processing workflow.

\- Record the version and environment used.



\### 4.5 Python Module Import Error



\*\*Cause:\*\* A module import may refer to a filename or package path that does not match the actual project structure.



\*\*Corrective actions:\*\*

\- Verify the module filename and its location.

\- Check package and import paths.

\- Confirm that the intended Python environment is active.

\- Rerun the relevant test after correcting the import.



\## 5. Validation Requirements and Limitations



The Phase III test file records 11 passing tests covering LiDAR point classification, slope-risk behavior, site-profile fields, input validation and walkability-score behavior. These tests demonstrate the tested functions' behavior under the included test cases; they do not establish that the complete GeoSense workflow has been validated.



The following work remains necessary:



1\. Validate LiDAR-derived metrics against independent reference data.

2\. Compare semantic classifications with manually labelled street-level imagery.

3\. Confirm data provenance, spatial coverage, temporal relevance and licensing.

4\. Verify environment reproducibility and document required software versions.

5\. Connect and test actual Phase I and Phase II outputs in the unified profile.

6\. Test the integrated workflow using compatible data from the same study area.

7\. Back up project files to Google Drive and an external drive.

8\. Review the repository status and update documentation after verification.



\## Conclusion



GeoSense Phase III extends the site-assessment workflow through LiDAR point-cloud analysis, street-view semantic analysis and structured JSON profile generation. The current implementation demonstrates these components and passes the recorded 11-test suite. Nevertheless, the different geographic coverage of the example datasets and the unavailable Phase I and Phase II outputs mean that the complete end-to-end assessment is not yet validated. These limitations must be addressed before the integrated workflow is considered complete.




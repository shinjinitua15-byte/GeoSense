\# GeoSense Phase III - Troubleshooting Guide



\## 1. Purpose



This guide documents common setup and execution problems that may occur while running the GeoSense Phase III LiDAR and street-view analysis modules. It provides diagnostic commands and safe troubleshooting steps for the verified Windows Python environment.



\## 2. Python Environment Problems



\*\*Problem:\*\* Python or required packages cannot be found.



\*\*Possible cause:\*\* The wrong Python environment is active, or dependencies are missing.



\*\*Diagnostic command:\*\*



```bat

D:\\GeoSense\_Phase3\\geosense\_lidar\\python.exe --version

```



\*\*Check installed packages:\*\*



```bat

D:\\GeoSense\_Phase3\\geosense\_lidar\\python.exe -m pip list

```



\*\*Recommended action:\*\*

\- Use the verified Python executable shown above.

\- Install project dependencies from the repository root:



```bat

python -m pip install -r requirements-phase3.txt

```



\- Install the appropriate PyTorch build separately, following the project's environment guide.

\- Avoid installing packages into a different Python environment by mistake.



\## 3. PyTorch and CUDA Problems



\*\*Problem:\*\* PyTorch cannot detect the GPU, or CUDA is unavailable.



\*\*Diagnostic command:\*\*



```bat

D:\\GeoSense\_Phase3\\geosense\_lidar\\python.exe -c "import torch; print(torch.\_\_version\_\_); print(torch.cuda.is\_available()); print(torch.cuda.get\_device\_name(0) if torch.cuda.is\_available() else 'CUDA unavailable')"

```



\*\*Recommended action:\*\*

\- Confirm that the installed PyTorch build is compatible with the intended CUDA setup.

\- Check GPU driver compatibility.

\- If CUDA is unavailable, review the error before changing or reinstalling packages.

\- Do not assume that GPU acceleration is working unless `torch.cuda.is\_available()` returns `True`.



The verified environment previously reported PyTorch `2.5.1+cu121` and CUDA availability as `True`. Other computers may have different configurations.



\## 4. Missing LiDAR Data or Empty Results



\*\*Problem:\*\* A LiDAR analysis returns no points or an empty result.



\*\*Possible causes:\*\*

\- The input LAS/LAZ file is missing or unreadable.

\- The requested bounding box does not overlap the available point cloud.

\- The input coordinates use a different coordinate reference system.

\- The requested study area is outside the available dataset coverage.



\*\*Recommended action:\*\*

\- Confirm that the input file exists and can be opened.

\- Check the point cloud's coordinate reference system and spatial extent.

\- Verify that the study-area coordinates and bounding box use compatible coordinate systems.

\- Record the input dataset, coverage, and processing parameters.



\*\*Known limitation:\*\* The previously tested LiDAR dataset did not cover Chennai. An empty Chennai LiDAR result should not be interpreted as evidence that buildings or vegetation are absent.



\## 5. Missing Model Checkpoint



\*\*Problem:\*\* The Segment Anything Model (SAM) cannot load its checkpoint.



\*\*Expected checkpoint path:\*\*



```text

D:\\GeoSense\_Phase3\\models\\saved\\sam\\sam\_vit\_b\_01ec64.pth

```



\*\*Recommended action:\*\*

\- Confirm that the checkpoint file exists at the expected location.

\- Check that the configured model type matches the checkpoint.

\- Verify that the checkpoint is complete and readable.

\- If the file is unavailable, obtain it from the authorized source and confirm its integrity before running the model.



Model checkpoints are separate from Python package dependencies and may not be included in the Git repository.



\## 6. Mapillary Access and Environment Variables



\*\*Problem:\*\* The Mapillary downloader reports a missing token or authentication failure.



\*\*Environment variable:\*\*



```text

MAPILLARY\_ACCESS\_TOKEN

```



\*\*Recommended action:\*\*

\- Confirm that the local `.env` file is present at the configured project location.

\- Check that the variable name is spelled correctly.

\- Confirm that the token is valid and has the required access permissions.

\- Check network connectivity and the API response if authentication still fails.



\*\*Security precautions:\*\*

\- Never commit `.env` files or access tokens to GitHub.

\- Never paste access tokens into public issues, screenshots, or shared logs.

\- If a token is exposed, revoke it and generate a replacement.

\- Do not print the token while debugging.



\## 7. Module Import Errors



\*\*Problem:\*\* Python reports `ModuleNotFoundError` or cannot import a Phase III module.



\*\*Recommended action:\*\*

\- Confirm that the command uses the intended Python environment.

\- Check that the required package is installed in that environment.

\- Run scripts from the expected project directory when local imports depend on the working directory.

\- Check the module filename and import statement.

\- Read the complete traceback before modifying code.



Example package check:



```bat

D:\\GeoSense\_Phase3\\geosense\_lidar\\python.exe -c "import laspy, numpy, pandas, torch; print('Core imports successful')"

```



\## 8. Running the Phase III Tests



From the repository root, run:



```bat

D:\\GeoSense\_Phase3\\geosense\_lidar\\python.exe -m pytest tests\\test\_phase3\_modules.py -v

```



\*\*If tests fail:\*\*

\- Read the name of the failing test and the complete traceback.

\- Separate dependency errors from data, path, and logic errors.

\- Confirm that required files and model checkpoints are available.

\- Record the failing test and error message before changing code.

\- Rerun the tests after making a targeted correction.



The previously verified test run passed 11 tests. This is a historical result, not a guarantee that every future run will pass.



\## 9. Path and File Errors



\*\*Problem:\*\* A script cannot find an input, output, configuration, or model file.



\*\*Recommended action:\*\*

\- Check whether the path exists.

\- Confirm that the path matches the current computer's directory structure.

\- Check for missing folders, spelling differences, and incorrect file extensions.

\- Prefer explicit paths during initial diagnosis.

\- Review absolute paths before moving the project to another computer.



Some Phase III modules use paths under `D:\\GeoSense\_Phase3`. Moving the project may require configuration changes.



\## 10. Interpreting Results Carefully



Successful execution does not automatically mean that every output is scientifically validated.



\- Point-cloud class percentages are proxies and should not be presented as direct land-area estimates.

\- Slope-risk calculations should be interpreted according to the implemented formula and its limitations.

\- Street-view semantic classifications require manual validation before being treated as reliable measurements.

\- Rooftop segmentation outputs depend on the selected model, prompts or settings, and image conditions.

\- LiDAR and Chennai street-view imagery represent different locations. They must not be combined as observations from the same site.

\- Passing software tests verifies the tested behaviours only; it does not validate every scientific metric or research conclusion.



\## 11. Reporting a Troubleshooting Issue



When documenting an issue, record:



1\. The command that was executed.

2\. The Python executable and package versions.

3\. The complete error message or traceback.

4\. The relevant input file and its availability.

5\. Whether the problem can be reproduced.

6\. The corrective action taken and the result of retesting.



Remove tokens, passwords, personal information, and other secrets before sharing logs.



\## 12. Maintenance and Status



This guide should be updated whenever a recurring issue is identified or the verified environment changes. Verify all commands and dependency versions against the actual environment before adopting them on another computer.



This document is a troubleshooting reference. It does not establish that every Phase III workflow, metric, or integration has been fully validated.


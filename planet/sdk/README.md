# SDK for classic matrix based stacking 

## Pipeline:

### 1) Feature engineering:
 - Input: .csv file
 - Output: pandas DataFrame object(s)
 - Remove bad rows and impute NaN or errneous values
 - Feature selection & elimination
 - Creating new columns or grouping columns into separate csv files

### 2) models:
 - Input: pandas DataFrame object
 - Output: pandas DataFrame object
 - model classes that implement k-fold training and prediction as well as test-time training and prediction

### 4) Ouput formatting:
 - Input: pandas DataFrame object
 - Output: .csv file
 - Formats a pandas DataFrame object into a submittable file

### utils:
 - contains useful methods such as k-fold generation, file IO, etc

### main.py:
 - Boilerplate code and high level logic for running the pipeline

# DICOM to NIfTI Converter

This Python script converts a collection of DICOM files into a 3D NIfTI image while considering metadata from a Parquet file. It utilizes the `os`, `numpy`, `cv2`, `nibabel`, `pydicom`, `pandas`, and `multiprocessing` libraries.

## Prerequisites

Before using this script, ensure that you have the necessary libraries installed. You can install them using the following commands:

```bash
pip install numpy opencv-python-headless nibabel pydicom pandas
```

## Usage

1. Save the script in a Python file (e.g., `dicom_to_nifti_converter.py`).

2. Customize the script by specifying your input directory, output directory, Parquet file path, and the desired dimensions for the 3D volume:

```python
input_dir = '/path/to/your/DICOM/files'
output_dir = '/path/to/your/output/directory'
parquet_file = '/path/to/your/parquet/file.parquet'
x_size, y_size, z_size = 128, 128, 128
```

3. Run the script in your terminal or command prompt:

```bash
python dicom_to_nifti_converter.py
```

The script will process the DICOM files in your specified input directory, create 3D volumes, and save them as NIfTI files in the specified output directory.

## Notes

- Make sure the input directory contains the DICOM files you want to convert.
- You should have the Parquet metadata file that corresponds to the DICOM files.
- The script will create a directory structure in the output directory to organize the converted NIfTI files.

## Example

Here's an example of how to use the script:

```bash
Processing patient <patient_id>, scan <scan_id>...
Creating 3D volume...
Saving the 3D volume...
Combined and resampled image saved as '<output_file>'.
```

Enjoy converting your DICOM files into 3D NIfTI images with metadata consideration!

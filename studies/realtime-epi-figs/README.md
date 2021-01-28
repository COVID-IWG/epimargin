# *Nature Research* Code & Software Submission Checklist 

This directory contains the content required for submissions to the Nature Research family of journals.

1. System requirements:
- *Software dependencies*: Python 3.6+, all operating systems
- *Versions the software has been tested on*: 
  - Python 3.6
  - Python 3.7
  - Python 3.8
  - macOS Mojave (10.14)
  - macOS Catalina (10.15)
  - Windows 10
  - Scientific Linux 7.4 (Nitrogen)
  - Ubuntu 20.04.1 LTS (Focal Fossa)
- *Any required non-standard hardware*: none

2. Installation guide
- *Instructions*: The `adaptive-control` project is a Python3 package, and can be installed by following these steps:

  A. Install Python3 on your development machine, if you do not already have it.

  B. Clone or download the `adaptive-control` repository.

  C. Within the repository folder, create a virtual environment.
  ```
  python3 -mvenv venv
  ```

  D. Activate your virtual environment.
  ```
  source venv/bin/activate
  ```

  E. Install the requirements.
  ```
  pip install -r requirements.txt
  ```

  F. Install the project. 
  ```
  pip install -e . 
  ```

- *Typical install time*: < 5 minutes

3. Demo 
- *Instructions to run on data*: for any Python file in this 
  ```python3 <name of file>```
- Expected output: 
  - plots for most files, and regression tables for `regressions.py`
- Expected run time for demo: < 5 minutes

4. Instructions for use
- *How to run the software on your data*: see step 3

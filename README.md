==================== HMM-Based SASI Frameworks ==================== 

This repository contains the Python source code and experimental datasets 
used in the study presented in [1].
__________________________________________________________________________________________________
   I- Repository Structure
   
   The repository is organized into the following main subdirectories:
   
   1- RFM representations
   
   This directory contains the implementations and experimental data related 
   to the RFM-based representations investigated in [1].
   
   2- SASI frameworks
   
   This directory contains the implementations and experimental data related 
   to the SASI frameworks proposed and evaluated in [1].
__________________________________________________________________________________________________
   II- REQUIREMENTS
   
   All experiments reported in [1] were conducted under the Microsoft Windows 
   operating system. 
   
   Before running the programs available in this repository, the following 
   software must be installed.

   1- Python 3.9.11
   
   All Python programs were developed and tested using Python 3.9.11.
   
   Before executing any script, please inspect the corresponding .py file and 
   install all required Python packages..

   * Important Note About "tslearn" *
   
   To execute the program ".\RFM representations\dynamic RFM\drfm.py" without 
   affecting your global Python installation, it is strongly recommended to create 
   a dedicated virtual environment before installing the package "tslearn".
   Additional installation instructions are provided at the beginning of the 
   corresponding Python file.

   2- Weka 3.9.3
   
   Weka 3.9.3 was used for all K-means clustering experiments reported in [1], 
   except the DTW-based clustering experiments, which were implemented directly 
   in Python.
    
   3- Java 11.0.30
   
   Java is required to execute Weka command-line operations from the provided 
   .bat files.     
__________________________________________________________________________________________________
   III- How to Run the Programs?
   
   Each final subdirectory contains a command file named: "run_program.bat".
   Executing this file launches the corresponding experiment or processing pipeline.
   
   In some directories, the .bat file requires input parameters. These parameters are 
   documented as comments (lines beginning with ::) at the beginning of the batch file.
   
   Typical parameters include:
   
   - the number of clusters K for the K-means algorithm,
   - or the full path to the "weka.jar" file on the target machine.
   
   For the experimental environment used in [1], the Weka library was located at:
   "C:\Program Files\Weka-3-9\weka.jar"
_________________________________________________________________________________________________
    IV- Reference
	
[1] SYLVAIN ILOGA, and K. M. MOTUE DJOKO, "The SASI frameworks: Efficient HMM-based alternatives to 
    RFM representations for customer segmentation and customer profile analysis in the banking sector",
	(Under review).
__________________________________________________________________________________________________

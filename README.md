# Machine Learning
---
## Malfunctioning Industrial Machine Investigation and Inspection Using Sound

- Repository: `Malfunctioning_Industrial_Machine_Investigation_and_Inspection_Using_Sound`
- Duration: `2 weeks`
- Deadline: `04/02/2021 10:00 AM`
- 'Team challenge'
- Team Members:
	- [Bram De Vroey](https://github.com/brmdv)
	- [Sijal Kumar Joshi](https://github.com/sijal001)


# Mission objectives

* Create a Machine Learning model that predicts when a machine will fail, based on the current sound.
* Extra: Model that can categorize the failures. This help to do more targeted maintenance.


# Learning Objectives

* To be able to work in a team 
* To be able to complete task in given timeline
* To be able to understand business needs and problems
* To be able to present model to customer.
* To be able to present a final product.

# Data preprocessing
In order to convert sound files to useable data points, we had to extract a number of discrete feautures from each audio file. We used the Python library Librosa, which can be used to read and analyse audio. Librosa provides a multitude of processing functions, in the time domain, and predominently in the frequency domain. After some experimentation, we selected about ten features, most of them based on the MEL spectrogram of each audio file -- a representation that is commonly used for ML in audio.

Because most of this audio processing gives higher-dimensional results. decisions about how to collapse them into discrete numbers had to be made. Because of time restrictions, and because of promising results with the simple selection of mean, standard deviation and median (the latter we later dropped because of similarity to the mean) gave us good-enough results.
However, we are aware that we loose some information, and not all subtitlities will be characterized. This is a domain in which futher imporvements could be reached, for example by representing some features by other, more suitable statistical aggregations.

All the data preprocessing is done inside preprocessing.py, which holds functions to read the MIMII zip files, and apply Librosa processing on each wave file inside.


# The Mission

This dataset is a sound dataset for malfunctioning industrial machine investigation and inspection (MIMII dataset). It contains the sounds generated from four types of industrial machines, i.e. valves, pumps, fans, and slide rails. Each type of machine includes seven individual product models*1, and the data for each model contains normal sounds (from 5000 seconds to 10000 seconds) and anomalous sounds (about 1000 seconds). To resemble a real-life scenario, various anomalous sounds were recorded (e.g., contamination, leakage, rotating unbalance, and rail damage). Also, the background noise recorded in multiple real factories was mixed with the machine sounds. The sounds were recorded by eight-channel microphone array with 16 kHz sampling rate and 16 bit per sample. The MIMII dataset assists benchmark for sound-based machine fault diagnosis. Users can test the performance for specific functions e.g., unsupervised anomaly detection, transfer learning, noise robustness, etc.

---

# About Running the Program

* **Python version:** `3.8.8`

**Imporant Libaries:**

| Library       | Used to                                        |
| ------------- | :----------------------------------------------|
| numpy		|to work around multi-dimensional of generic data|
| os		|to work around system path.			 |
| matplotlib	|to genereate ploting.		                 |
| pandas	|to remove, move, copy files.			 |
| shutil	|to remove, move, copy files.			 |
| pickel	|to remove, move, copy files.			 |
| imblearn	|offering a number of re-sampling techniques.	 |
| warnings	|to remove, move, copy files.			 |
| sklearn	|Machine learning library for the Python. 	 |
| librosa 	|to analysis music and audio. 			 |


**Note:** Just use command `pip install -r requiement.txt` to install the required libary with correct version and run the program smoothly.




# **MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection**

* **Storage Requirement:** 100 GB 

* ***Method :*** Manually Download all the file from link save to the respective folder.
    * [Sound File Link](https://zenodo.org/record/3384388#.YGXS5ntR3-j)


# Repo Architecture

```
Malfunctioning_Industrial_Machine_Investigation_and_Inspection_Using_Sound
│
│   README.md               :explains the project
│   requirements.txt        :packages to install to run the program
│   .gitignore              :specifies which files to ignore when pushing to the repository
│__   
│  Data_Model_analysis      :directory contain all the main .ipynb that create the machine model that train test and creates a pickel files.
│   │
│   │ Fan_data_analysis     :notebook that contain data, Machine learning model, metric,statics, etc. fan.
│   │ Pump_data_analysis    :notebook that contain data, Machine learning model, metric,statics, etc. pump.
│   │ Slider_data_analysis  :notebook that contain data, Machine learning model, metric,statics, etc. slider.
│   │ Valve_data_analysis   :notebook that contain data, Machine learning model, metric,statics, etc. valve.
│   │
│   │ dataset		    ::directory contains all .pynb file that does the preprocessing and fearure extration .
│     │__
│	 processed_data     :directory contains the .csv files that contains the main machine features and information.
│	 preprocessing.py   :python script file to preprocess ad grab the features
│	 get_data.py        :python script file that read and manage all sound files
│__   
│  main		    	    :directory contain all the main .ipynb that create the machine model that train test and creates a pickel files.
│   │
│   │ Pump_ML_model.ipynb   :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine pump.
│   │ Slider_ML_model.ipynb :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine slider.
│   │ Fan_ML_model.ipynb    :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine fan.
│   │ Valve_ML_model.ipynb  :notebook that contain trainning, pickel creatin script and abnormal sound clusting for machine valve.
│   │ predict.py	    :end user script file, script to how user upload data and decide what model to use.
│   │
│   │ saved_model      	    :directory contains all saved pickel files of the machine learning model.
│   │ dataset		    ::directory contains all .pynb file that does the preprocessing and fearure extration .
│     │__
│	 processed_data     :directory contains the .csv files that contains the main machine features and information.
│	 preprocessing.py   :python script file to preprocess ad grab the features
│	 get_data.py        :python script file that read and manage all sound files
```

---

# Instruction
#### How to get Prediction

1. Setup python environment  `3.8.8`.
2. Install all libaries `pip install -r requirements.txt`.
3. Download important "sound data" files, generate `csv` using 'get_data.py'.
4. Run all `.ipynb` inside jupyter notebook to retrain and generate pickel files.
5. Run `predict.py` and fill the information as requested and get the prediction.
 
---
# Next Step

* Optimize the model with better data set and feature to genereate better predicition.
* Improving end user experiance.

## Project Overview

  

This project focuses on analyzing and predicting the lifespan of worms under various experimental conditions. The primary data consists of CSV files, each of which captures the x-y center-of-mass coordinates of individual worms over time. The project involves processing, feature extraction, and machine learning model training to predict lifespan-related outcomes for worms exposed to different drug conditions.

---

  
### **Data Structure**

The data is organized within the `./data/Lifespan` directory and follows the structure outlined below.

#### **Subfolders**
- **`companyDrug`**: Data from worms exposed to a specific drug.
- **`control`**: Data from worms without drug exposure.
- **`Terbinafin`**: Data from worms exposed to Terbinafin.
- **`controlTerbinafin`**: Data from worms without drug exposure.

> **Note:** The folder names listed above are **examples** and can be different. You can add additional subfolders to accommodate different drug conditions as needed.

#### **Data Recording Details**
- **Time Interval**: The time difference between consecutive frames is **2 seconds**.
- **Recording Duration**: Each recording session lasts for **900 frames** (30 minutes).
- **Session Gap**: There is a **5.5-hour gap** between consecutive recording sessions.
- **Frame Numbering**: Frame numbering restarts after **10799** to facilitate better organization and compatibility with the equipment.

### Setup Instructions

  

#### 1. Install Conda

Ensure that you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution) installed on your system.

  

#### 2. Create a Conda Environment

Use the provided `environment.yml` file to create a Conda environment named `animal-lifespan-prediction`:

  

```bash

conda  env  create  -f  environment.yml

```

  

#### 3. Activate the Environment

After the environment is created, activate it using:

  

```bash

conda  activate  animal-lifespan-prediction

```
#### 4. Run the Pipeline

To execute the data processing and feature extraction, run the following command in the appropriate environment:

```bash

python run_pipeline.py --base-dir ./data --config pipeline_config.json

```

**Note:** Ensure that the folder structure is properly aligned with the "Data Structure" outlined above.

---

#### 5. Train the Model

To train the model using the processed data, run the following command:

```bash

python train_model.py --input ./data/Lifespan_features --config train_model_config.json

```

---

#### **Important Note**

Ensure that the JSON configuration file `train_model_config.json` contains the correct subfolder names under the `subdirs` key within the `data` object. These subfolder names must align with the actual folder names present in the `./data/Lifespan` directory.

**Example JSON configuration:**

```json

"data": {

"base_dir": "data/Lifespan_features",

"subdirs": ["control", "Terbinafin", "controlTerbinafin", "companyDrug"]

}

```

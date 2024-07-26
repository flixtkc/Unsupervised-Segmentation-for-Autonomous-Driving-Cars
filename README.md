# USAD

This repository contains the implementation of an unsupervised segmentation model specifically designed for autonomous driving applications. The goal of this project is to develop a robust segmentation algorithm that can operate without labeled data, providing a scalable solution for real-world autonomous driving scenarios.

The project consists of multiple key sections in order to perform the semantic segmentaton. Below you see the general steps outlined to get a rough idea

#### Data Collection

#### Data Processing To Desired Format

#### Cropping Utility To Improve Spatial Resolution

#### Precomputation of KNN indices

#### Training of Model

#### Evaluation

#### Real-Time Segmentation or Video/Image Segmentation


### Code source

This repository contains code from other sources
- Modified:
  <!-- - [World on rails](https://github.com/dotchen/WorldOnRails) -->
  - [Cheating by Segmentation 2](https://github.com/maelwildi/CBS2) (branch: cbs2)
  - [STEGO](https://github.com/mhamilton723/STEGO/tree/master)

- Not modified/implemented:
    - [DriveAndSegment] (https://github.com/vobecant/DriveAndSegment)


### Installing Carla

Install Carla in the  desired location:
```
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
tar -xvzf CARLA_0.9.10.1.tar.gz -C carla09101
```
Clone USAD repository in desired location:
```bash
# Clone the repository
git clone https://github.com/flixtkc/Unsupervised-Segmentation-for-Autonomous-Driving-Cars.git

# Navigate into the project directory
cd Unsupervised-Segmentation-for-Autonomous-Driving-Cars
```
For this project there are two distinct environments you need to create. If not already done, install conda.
```bash
cd CBS2
conda env create -f docs/cbs2.yml
conda activate cbs2
```
Once tested to see if the environment is setup correctly, continue with the second environment (first go back to main directory).
```bash
cd ..
cd STEGO
conda env create -f environment.yml
conda activate stego
```

Add the following environmnet variables to `~/.bashrc`:
```bash
export CARLA_ROOT=<your_path>/carla09101
export CBS2_ROOT=<your_path>/CBS2
export LEADERBOARD_ROOT=${CBS2_ROOT}/leaderboard
export SCENARIO_RUNNER_ROOT=${CBS2_ROOT}/scenario_runner
export PYTHONPATH=${PYTHONPATH}:"${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
```
Verify the setup by launching Carla (with cbs2 virtual environment activated):
```bash
source ~/.bashrc
$CBS2_ROOT/scripts/launch_carla.sh 1 2000
```

### Setup Configurations

To set up configurations for the data collection there are several files to consider. Open any text editor of your choice and inspect the following file
```bash
CBS2/autoagents/collector_agents/config_data_collection.yaml
CBS2/autoagents/collector_agents/collector.py
CBS2/rails/data_phase1.py
CBS2/assets/
```
##### config_data_collection.yaml:

This file contains all the settings that define the data collection phase. Especially the resolution settings are key in determining how well the segmentation model will perform.
##### collector.py:

This file specifies all the collector functions. It also includes configurations for logging to Weights & Biases (wandb), which is set to False by default.

##### data_phase1.py:
The former denoting all the settings that define the data collection phase. The second one specifies all the collector functions, where also the logging to wandb can be found (default False). The latter containing the settings to set the driving episode specific config
This file contains settings for configuring specific driving episodes.

##### assets/ directory:

Here you can find the routes and scenarions for each town, which are needed to run any data collection succesfully. You need to specify which one to use in data_phase1.py


### Data Collection

Make sure to start Carla in one terminal, then boot up a second screen/terminal, go to the CBS2 subdirecory there run the data collection script:
```bash
# Terminal 1:
$CBS2_ROOT/scripts/launch_carla.sh <num_runners> <port>

# Terminal 2:
cd CBS2
python rails/data_phase1.py --port <port> --num-runner=<num_runners>
```
Note: if there are multiple runners, `<port>` is also the increment between them.


### Data Processing To Desired Format

After you have verified the collected data and are content with the results, you can continue to the data conversion step. Where the saved data has to be processed before the STEGO model can effectively train with it. Go back to the main directory where you can find the lmdb_to_STEGO_dataset converter script. Inspect the possible argument passed to the functions before running this script. NOTE: the dataset input path and output path need to be specified

#### Switch Environments
After this step it is important to deactivate cbs2 and activate the stego environmetn, as the next steps will need that adjustment.

### Configurations for STEGO

Inspect the configuration for the STEGO related script in:
```bash
cd STEGO/src/
vi configs/train_config.yml
```
Specify the dataset path, which is identical to the output path used in the previous step. For the next step you need to adjust the hyperparameters related to the cropping. As these heavily influence memory related errors.

### Cropping Utility
```bash
has_labels: False
crop_type: "five"
crop_ratio: .5
res: 200
loader_crop_type: "center"
```
The crop_type determines whether it takes 5 crops one from each corner and then on in the middle, or, random crops from the input image(s). Now the crop ratio determines how big/small the resulting crops will be compared to the input dimensions of the image. Tweak until you have favorable settings. After cropping please adjust the dataset path in the config file to point to the newly created cropped dataset directory

### Precomputation KNN indices
To speed up training steps it is crucial to run a precomputation of KNN indices 
```bash
python precompute_knns.py
```

### Train STEGO Model
Now all that is left is to run a training script on your custom data to see the segmentation results of the STEGO model on the collected  Carla simulator data. Please specifiy in the config file the logs directory!
```bash
pyton train_segmentation.py
```
During training you can monitor the entire phase by running a tensorboard session in the specified logs directory to see some intermittent progress.
```bash
# Example
tensorboard --logdir logs/logs/five_crop_0.5/directory_new_crop_date_Jul25_02-25-32/default/version_0/
```

### Evaluation and Testing
To evaluate and compare your training results you can monitor tensorboard logs of course where you can see various metrics over time, but you can also run a real-time segmentation script to see a  real-life application. As discussed in the the thesis this resulted in no real-time application, but will be improved upon in the future. Additionally, there is also a script that takes a normal video and segments it for you given the specified model, and then saves the video. You can uncomment the image or video segmenter, whichever one you prefer. For an example please see the testing_videos/ directory!

```bash
cd STEGO/src/
python STEGO_create_segmented_video_or_image.py
python STEGO_real_time_segmenter.py
# See
cd ../../testing_videos/
```

### Extra tests
Additionally, there are several scripts to test your systems batch size, num workers and ssh X11 forwarding in case you run into errors. 


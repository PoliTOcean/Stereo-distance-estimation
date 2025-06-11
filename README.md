# Stereo-distance-estimation
Interactive interface to estimate length using stereo camera and SuperGlue model.

## Setup the Environment
Create Python environment
```bash
python -m venv env
```
Activate the environment 
```bash
source env/bin/activate
```
Install requirements
```bash
pip install -r requirmenets.txt
```

## Instruction Manual
Run the script
```bash
python super_glue.py
```
For the first part you need to select between 2 and 4 points on the image using the mouse leftclick, if you want to deselect your last point use mouse rightclick. When you are done with the selection press ESC to go to the next stage. You need at least 2 points, if not enough points are selected you need to do this phase again.

Then the script will match your selected points with the keypoints on the image and it will compute the disparity. If a match is not found or the computation of the disparity fails, you need to select your points again. 

The third step consists of chosing if the distance is computed using the single point or the middlepoint of twos. You need to make 2 choices: press 's' if you want to use the first (or third) point as first edge or press 'm' if you want to use the middle point of the first and second selection (or the third and forth selection).

After this step the script will compute the lenght distance between the two edge points, the value is printed on the terminal. 

At every step information are available on the terminal. 



# Posture-Classification-Using-Pose
The aim of the project was to study the possibilities of identifying good or bad postures of a person performing exercise. As a particular use case, we chose the exercise deadlift and created a small dataset of images collected over the internet. The image is processed using a pre-trained pose estimation model and the output pose is later used by another model to classify the pose as good or bad.


<div style="text-align:center">
  <img src="./assets/architecture.png" alt="Architecture">
</div>

To run the output, you can install the required libraries.
```
pip install -r requirements.txt
```

The dataset can be downnload using this [link](https://drive.google.com/drive/folders/1w5JttdBhfla05oVcg0Spd6eg_Zaq5cI4?usp=sharing). The pose GT for the images were obtained by initially running "Keypoints_rcnn" and correcting them when required using a GUI. The code for the GUI and data preparation is not provided here. To train the model, run

```
python keypoint-pose-train.py
```
For further details, refer the "Report.pdf"

## References
1. Chen, K.: Sitting posture recognition based on openpose. IOP Conference Series:
Materials Science and Engineering 677 (2019)
2. Human Pose Estimation using Keypoint RCNN in PyTorch, [link](https://learnopencv.com/human-pose-estimation-using-keypoint-rcnn-in-pytorch/)
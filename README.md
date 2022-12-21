# UAVID-Dataset-Semantic-Segmentation

This project aims to create a semantic segmentation model by training with the uavid dataset. The classes to be distinguished are:


<ul>
  <li>Building</li>
  <li>Road</li>
  <li>Static car</li>
  <li>Tree</li>
  <li>Low-vegetation</li>
  <li>Human</li>
  <li>Moving car</li>
  <li>Background clutter</li>
</ul>

In order to convert 3 channel label image to 1 channel integer labels [UavidToolKit](https://github.com/YeLyuUT/UAVidToolKit) repository is used. After converting labels to one hot encoding form model is trained.

After prediction there is visualization of predicted labels with real image.

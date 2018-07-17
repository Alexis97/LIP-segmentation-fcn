# LIP-segmentation-fcn
* Train a simple FCN to do segmentation with LIP dataset.
* The FCN network structure is from [MarvinTeichmann/tensorflow-fcn](https://github.com/MarvinTeichmann/tensorflow-fcn). Thanks a lot for this simple but effective work!
* The train dataset I use is [Look into Person](http://www.sysu-hcp.net/lip/). 
* Here are some demo results:
<center>
	<img src="https://github.com/Alexis97/LIP-segmentation-fcn/blob/master/demos/image/image_58000_2.png" height = 200>
	Image resize to 384x384
</center>

<center>
	<img src="https://github.com/Alexis97/LIP-segmentation-fcn/blob/master/demos/label/label_58000_2.png" height = 200>
	Label resize to 384x384
</center>

<center>
	<img src="https://github.com/Alexis97/LIP-segmentation-fcn/blob/master/demos/predict/predict_58000_2.png" height = 200>
	Predict result of 384x384 after 58000 steps
</center>
## Usage


# NCFM fish recognition #

### To do: ###

* use [this guy's labels] (https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/25565/annotated-mouth-and-tail) to train a CNN to detect points on a fish
* apply affine transformation to the image and crop
* gamma balance and color balance (anything to do with image standardization i.e. linearlization and normalization)
* train a vgg16 network on labeled, processed training data 

#### Other ideas: ####
* account for the fact that some images are night / day (RGB color normalization?)
* flip right side up
* use information about similar [boats] (https://www.kaggle.com/anokas/the-nature-conservancy-fisheries-monitoring/finding-boatids)?
* multiple fishes
* sharpening
* perspective (non-affine transforms)
* augment training set: RGB color shift, random scaling, random perspective warp, instead of flipping all images right side up (since this might be imperfect), flip all images vertically so there is exactly 50% right side up, 50% upside down
* use smaller neural networks due to training set being relatively small (vgg16, zf, etc)

### Useful Repos: ###
* https://github.com/rdcolema/nc-fish-classification
* https://github.com/pengpaiSH/Kaggle_NCFM

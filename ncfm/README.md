# NCFM fish recognition #

### Labeling: ###
YFT (yellowfin tuna) :
  * 187 images (not including duplicates)
  * classififed up to img_06769.jpg 
  * IN DECENDING ORDER
  * mahi mahi in images 06971 and 06763
SHARK:
  * 101 images
  * not all labeled, didn't label duplicates past img_04819.jpg
  * IN DESCENDING ORDER
LAG (moonfish, opah):
  * 55 images
  * not all labeled due to duplicates
DOL (mahi mahi):
  * 112 images
  * most labeled minus a few poor quality images
BET (Bigeye tuna):
  * 182 images
  * most labeled minus a few redundant and poor quality images
ALB (Albacore tuna):
  * 204 images
  * most labeled minus some bad quality images
  * up to 04852 (desc)


### To do: ###
#### Michael: ####
* automate method to split between train val better
* check if validation augmentation has any point
* make validation error and training error closer to test error
* train more on images producing more error
* try identifying boats
* tune hyperparameters
* blend models

#### Ryan: ####
* gamma balance and color balance (anything to do with image standardization i.e. linearlization and normalization)
* dropout
* hard negative mining
* SIFT feature engineering
* try an R-CNN or SSD
* decreasing learning rate

### Other ideas: ###
* account for the fact that some images are night / day (RGB color normalization?)
* flip right side up
* use information about similar [boats] (https://www.kaggle.com/anokas/the-nature-conservancy-fisheries-monitoring/finding-boatids)?
* use the fact that there are sometimes multiple fish in a picture
* sharpening
* perspective (non-affine transforms)
* augment training set: RGB color shift, random scaling, random perspective warp, instead of flipping all images right side up (since this might be imperfect), flip all images vertically so there is exactly 50% right side up, 50% upside down
* use [this guy's labels] (https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/forums/t/25565/annotated-mouth-and-tail) to train a CNN to detect points on a fish


### Completed ###
* basic vgg16 implemented with keras
* added keras data augmentation for train/test
* clip probabilities to minimize loss
* save regular submission csv and modify when reading in/out

## Useful Repos: ##
* https://github.com/rdcolema/nc-fish-classification
* https://github.com/pengpaiSH/Kaggle_NCFM

#Ultrasound Nerve Segmentation#

###Summary###
The approach to this is easy and the implementation __should__ be easy, but dataset formatting will be a pain in the butt. We can use [FRCNN](https://github.com/rbgirshick/py-faster-rcnn) (Faster Region-based Convolutional Neural Network) and fine tune this with the given dataset to acheive high recognition rates. The most clever modification which doesn't take too much work is to experiment with different types of detection proposal methods and find the one that works best with ultrasound images. I wrote a fair amount of code to make Girshick's FRCNN code actually able to be applied to other datasets, which I can give you access to if you want, otherwise we can just see this as a black box. Pretty much, what we need to do is convert the given labels (masks) and convert from their format into bounding boxes in the Pascal VOC .XML format. Then, generate region proposals in a single .mat file. Then, since the output are also bounding boxes, we need to find a way for the output to be accurate polygons rather than rectangles. 

To Do:
- [ ] Format masks to .XML
- [ ] Determine best region proposal algorithm
- [ ] Figure out a way for accurate detection

# Urban land use mapping in data scarce regions 

The project consists of three parts: 
1. Open Street Map - processing OSM file to extract data from the ROI and reclassify labels from POI, building, and area object into urban land use classes.
2. Sampling ground truth - using the OSM generated shapefiles, satellite data is sampled. Building footprint statistics are calculated and sampled as well.
3. Deep learning - training and testing data is used to train a densenet model in order to classify urban land use.

(This just a first version of the code and will be updated once the paper is published.)  

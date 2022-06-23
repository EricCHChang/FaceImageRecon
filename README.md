# Matlab Codes For Face Image Reconstruction

**Contact:** Chi-Hsun (Eric) Chang (ch.chang@mail.utoronto.ca)

**Important Note:** These codes are for the use of academic research only. If you wish to use these codes in your research, please contact me first. 


## Preparation

### Behavioral data input
* A participant's behavioral responses (e.g., dissimilarity ratings) for all pairs of faces. The participant's pairwise ratings must be stored in a n-by-n matrix, where n is the number of face images. More specifically, the diagonal elements are 0 and the off-diagonal elements are the dissimilarity ratings of pairs of faces. Note that the dissimilarity score must be scaled to a range between 0 and 1 – 0 means identical or very similar and 1 means the most different/dissimilar. 
* If your participants rated similarity between faces rather than dissimilarity in the experiment, you must convert their responses to dissimilarity scores first

### Non-decomposed image reconstruction
* A cell array storing all face images (as RGB pixel intensity values) in which each cell stores an image

### Shape reconstruction
* A matrix storing the coordinates of 82 fiducial points for each face image (in the pipeline, the matrix is assigned to ```info_unfam_shape.mat```)
* A cell array storing all face images (as RGB pixel intensity values) in which each cell stores an image (this is not required if you are not constructing heatmaps of differences in accuracy for each fiducial point)

### Surface reconstruction (or non-decomposed reconstruction)
* A cell array storing all surface face images (as RGB pixel intensity values) in which each cell stores an image

### Path/Directory
* You may need to change the codes regarding paths and directories based on how you organize your data/files on your computer


## Perform Image Reconstruction

### Non-decomposed face image reconstruction 
* Run ```Pipeline_nonDecmp_ips.m```

### Shape reconstruction 
* Run ```Pipeline_shape_ips.m```

### Surface reconstruction
* Run ```Pipeline_surface_ips.m```


## Citation
* For **non-decomposed image reconstruction**, please cite: Chang, C. H., Nemrodov, D., Lee, A. C., & Nestor, A. (2017). Memory and perception-based facial image reconstruction. *Scientific reports, 7*(1), 1-9. https://doi.org/10.1038/s41598-017-06585-2
* For **shape and surface reconstruction**, please cite: Chang, C.-H., Nemrodov, D., Drobotenko, N., Sorkhou, M., Nestor, A., & Lee, A. C. H. (2021). Image reconstruction reveals the impact of aging on face perception. *Journal of Experimental Psychology: Human Perception and Performance, 47*(7), 977–991. https://doi.org/10.1037/xhp0000920

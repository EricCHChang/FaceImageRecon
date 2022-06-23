function outIm=remakeImage(inMat,bck,ims)
%% This function converts vector form of the image (or p-values) to 3 channel
% rectangular form and adds background.
%   Input:  inMat -     vector to convert: pixels X dims
%           bck -       indices of the background
%           ims -       cell with images
%   Output: outIm -     image: rows X columns X color chans X dims        
sizeIm=size(ims{1});
mat=zeros(1,bck(end));
mat(bck)=0.5;
mat=repmat(mat',1,size(inMat,2));
mat(mat==0)=inMat;
mat=reshape(mat,sizeIm(1),sizeIm(2),3,size(inMat,2));
outIm=mat;



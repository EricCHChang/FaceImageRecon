function [out_mat]=heat_map_recon(recon_mat,ims,chanNum)
%% Computing heatmap
%Inputs:
%       recon_mat=reconstructed matrix (row x col x channels x images)
%       ims - origional images in LAB space
%       chanNum - 3: all channels; 1: L channel
%Output:
%       out_mat - reconstruction accuracy for each pixel. The meanings of
%                 dimension are: number of images x number of pixels in y x
%                 number of pixels in x

if nargin<3 
    chans=1:3;
else
    chans=chanNum;
end

imm=NaN([size(recon_mat,1) size(recon_mat,2) size(recon_mat,3) length(ims)]);
%dimension in imm: number of pixel in y, number of pixel in x, color
%channel, number of images
for i=1:length(ims)
    imm(:,:,:,i)=ims{i};
end
for i=1:size(recon_mat,4)
    rec_mat_temp=repmat(recon_mat(:,:,chans,i),1,1,1,length(ims)-1);
    mat_true_diff=(recon_mat(:,:,chans,i)-ims{i}(:,:,chans)).^2;
    mat_true_diff=repmat(mat_true_diff,1,1,1,length(ims)-1);
    inds=setdiff(1:size(recon_mat,4),i);
    mat_other_diff=(rec_mat_temp-imm(:,:,chans,inds)).^2;
    mat_comp(i,:,:,:,:)=mat_true_diff<mat_other_diff;
end
out_mat=squeeze(mean(mat_comp,5));
out_mat=squeeze(mean(out_mat,4));


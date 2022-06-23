function [pos_mat,neg_mat]=protoPresent(loadings,ims,dims,norm,vis)
% this function shows prototype of the selected dimensions of MDS.
% Input:    loadings - loading matrix without leave-one-out;
%           ims - images as a cell array
%           dims - desired dimensions
%           norm - should the data be normalized in terms of brightness: 1
%                   - yes; 0 - no;
zload=zscore(loadings,1);
im_mat=NaN(size(ims{1},1),size(ims{1},2),size(ims{1},3),length(ims));
for i=1:size(ims,1)% converting image cell array to 2D array
    %im_mat(:,:,:,i)=double(ims{i});
    im_mat(:,:,:,i)=ims{i};
end
% ind_pos=find(zload>0);
% ind_neg=find(zload<0);
pos_mat=NaN([size(ims{1}) length(dims)]);
neg_mat=NaN([size(ims{1}) length(dims)]);
for i=1:length(dims)
    ind_pos=find(zload(:,dims(i))>0);
    ind_neg=find(zload(:,dims(i))<0);
    im_mat_pos=im_mat(:,:,:,ind_pos);
    zloadpos = permute(zload(ind_pos,dims(i)),[2 3 4 1]);
    Y_pos_mat=repmat(zloadpos, size(ims{1},1),size(ims{1},2),size(ims{1},3), 1);
    prot_pos=mean(im_mat_pos.*Y_pos_mat, 4);
    pos_mat(:,:,:,i)=prot_pos;
    
    im_mat_neg=im_mat(:,:,:,ind_neg);
    zloadneg = permute(zload(ind_neg,dims(i)),[2 3 4 1]);
    Y_neg_mat=repmat(zloadneg, size(ims{1},1),size(ims{1},2),size(ims{1},3), 1);
    prot_neg=mean(im_mat_neg.*Y_neg_mat, 4);
    neg_mat(:,:,:,i)=abs(prot_neg);
end

for i=1:length(dims)
    if norm
        neg=neg_mat(:,:,:,i);
        neg=neg/max(neg(:))*255;
        neg_mat(:,:,:,i)=neg;
        pos=pos_mat(:,:,:,i);
        pos=pos/max(pos(:))*255;
        pos_mat(:,:,:,i)=pos;
    else
        neg=neg_mat(:,:,:,i);
        pos=pos_mat(:,:,:,i);
    end
    
    if vis
        figure
        suptitle(['Prototype: Dimension ' num2str(dims(i))]);
        subplot (1,2,1)
        imshow(uint8(neg))
        
        subplot (1,2,2)
        imshow(uint8(pos))
    end
    
end

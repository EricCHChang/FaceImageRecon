function [pIds,threeValueMat]=threshThreeValueTrans(CI,p_mat,q,threshNum)
%% This function creates colored images for each LAB color channel:
%   Inputs:     CI -        image, source of polarity 
%               p_mat -     matrix with p-values
%               q -         q value for FDR correction
%               threshNum - minimum number of significant pixels
%   Outputs:
%               pIds -      matrix that shows where pID values were obtained:
%                           channels X dimensions
%               threeValueMat - images with three colors: grey for
%                           non-singificant, red-green for second channel
%                           and blue-green for third channel.
%   Example:    
% CI=CI_neut;
% p_mat=p_neut60;
% q=0.05;
% thresNum=10;
pIds=NaN(3,size(p_mat,2));
threeValueMat=NaN(size(p_mat));
chanSize=size(p_mat,1)/3; % number of pixels in one colour channel

for j=1:3
    rangChan=(j-1)*chanSize+1:j*chanSize;
    for i=1:size(p_mat,2) % number of dimensions
        [pID,~,~,~] = FDR_comp(p_mat(rangChan,i), q);
        if size(pID)>0
            p_temp=p_mat(rangChan,i);
            if sum(p_temp<pID)>threshNum % the number of p-values that are smaller than the FDR corrected threshold (pID) is larger than the minimum number of significant pixels
                threeValueMat(rangChan,i)=0.5;
                pval_vect_thr=double(p_mat(rangChan,i)<pID);
                CI_pol=double(CI(rangChan,i)>0);%polarity
                threeValueMat(rangChan,i)=threeValueMat(rangChan,i).*(1-pval_vect_thr)+(pval_vect_thr.*CI_pol);
                pIds(j,i)=pID;
            end
            
        else
        end
    end
end



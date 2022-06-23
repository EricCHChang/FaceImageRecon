function [pIds,CI_mat]=disp_CIs(CI_mat,p_mat,q_va,minPixThresh,bck,ims,numdims)
%% This function builds and presents significant CIs
% Input:    CI_mat -        Matrix with CIs: pixels X dims
%           p_mat -         Matrix with p values: pixels X dims
%           q_va -          q value for FDR: scalar
%           minPixThresh -  Required minimum number of pixels per channel: scalar 
%           bck -           index with background: vector
%           ims -           oringinal image cell array: cell
%           numdims -       number of dimensions to present: scalar
% Output:   pIds -          mat with signficance values: channels (3) X
%                           dims
%
if nargin<7
    doMinNum=1;
else
    doMinNum=0;
    if numdims>size(CI_mat,2)
        error('Number of dimensions is greater than number of dimensions in data')
    else
    end
end


[pIds,threeValueMat]=threshThreeValueTrans(CI_mat,p_mat,q_va,minPixThresh);
threeValueMat=remakeImage(threeValueMat,bck,ims);
CI_mat=remakeImage(CI_mat,bck,ims);
CI_mat=fixLab(CI_mat);
if doMinNum
    temp=max(pIds);
    temp=~isnan(temp);
    temp=find(temp);
else
    temp=1:numdims;
end

for i=temp
    figure
        %suptitle([kind ': Dimension ' num2str(i)]);
        suptitle(['Dimension ' num2str(i)]);
        subplot (3,2,1)
        imshow(CI_mat(:,:,1,i)) % raw classification image in L* channel
        subplot (3,2,2)
        if ~isnan(pIds(1,i))
            imshow(threeValueMat(:,:,1,i));
        else
            imshow(ones(size(CI_mat,1),size(CI_mat,2))*0.5);
        end
        subplot (3,2,3)
        imshow(CI_mat(:,:,2,i)) % raw classification image in a* channel
        subplot (3,2,4)
        if ~isnan(pIds(2,i))
            imshow(colorImageByChan(threeValueMat(:,:,2,i),2));
        else
            imshow(ones(size(CI_mat,1),size(CI_mat,2))*0.5);
        end
        subplot (3,2,5)
        imshow(CI_mat(:,:,3,i)) % raw classification image in b* channel
        subplot (3,2,6)
        if ~isnan(pIds(3,i))
            imshow(colorImageByChan(threeValueMat(:,:,3,i),3));
        else
            imshow(ones(size(CI_mat,1),size(CI_mat,2))*0.5);
        end
    
end

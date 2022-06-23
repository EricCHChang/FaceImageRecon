% find common background (mask) between images
function [outMask_threeChan,bck_common] = commonbck(ims_cell,doSym)
%Arguments
%       ims_cell - cell of image array (note this is in cell structure)
%                  each image is stored in a cell
%       doSym - whether force the surrounding to black and symmetrize it
%Output
%       outMask_threeChan - the background mask (3 colour channels)
%       bck_common - index of background (black pixel, value = 0)

%%
if nargin<2
    doSym = 1;
end

%%
for i = 1:length(ims_cell)
    img = ims_cell{i,1};
    img_mean = mean(img,3);
%     img_R = img(:,:,1); % pixels in 1st colour channel
%     img_G = img(:,:,2); % pixels in 2nd colour channel
%     img_B = img(:,:,3); % pixels in 3rd colour channel
%     img_sumRGB = img_R + img_G + img_B;
%     img_bck = find(img_mean==0); % index of black(0,0,0) pixel - zero in all 3 colour channels
    img_vec_mat(:,i) = img_mean(:);     
end

% product of pixel across all images to find out the maximum common
% background
% for example, if a pixel in any one of the images is zero (black), then it
% should be included in the common background
multiplyAll = prod(img_vec_mat,2); 
index_nonblack = find(multiplyAll);
multiplyAll(index_nonblack,1) = 1; % make non black pixel as 1 (white)

% reshape the vector to 2D image
multiplyAll_reshape = reshape(multiplyAll, [size(ims_cell{1,1},1), size(ims_cell{1,1},2)]);

% Forcing the surrounding to black and symmetrize it
temp = multiplyAll_reshape;
temp(1:2,:) = 0;
temp(end-1:end,:) = 0;
temp(:,1:2) = 0;
temp(:,end-1:end) = 0;
temp_mask = ones(size(temp));
for i = 2:size(temp,1)-1
    for j = 2:size(temp,2)-1
        if temp(i,j)==0
            temp_mask(i-1:i+1,j-1:j+1) = 0;
        end
    end
end
outMask = temp_mask.*fliplr(temp_mask);
if doSym
    outMask_threeChan = repmat(outMask,[1 1 3]);
else %if not making surrounding to be black and symmetrize
    outMask_threeChan = repmat(multiplyAll_reshape,[1 1 3]);
end

% Index of background (black)
bck_common = find(outMask_threeChan==0);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Below are obselete
% % Transform each image to a vector and put them together as a matrix
% % img_vec_mat: row: pixel; col: image
% for i = 1:length(ims_cell)
%     img = ims_cell{i,1};
%     img_vec_mat(:,i) = img(:);
% end
% 
% % Find out the common background across all images (mean=0)
% img_vec_mat_mean = mean(img_vec_mat,2);
% 
% % Make non black pixel as 1 (white)
% index_nonbck = find(img_vec_mat_mean);
% img_vec_mat_mean(index_nonbck,1) = 1;
% 
% % Reshape the vector to image size plus 3 colour channels
% img_vec_mat_reshape = reshape(img_vec_mat_mean,[size(ims_cell{1,1},1), size(ims_cell{1,1},2), size(ims_cell{1,1},3)]);
% 
% % % Index of background (black)
% % % (not forcing the surrounding to black_
% % outMask = img_vec_mat_reshape;
% % bck_common = find(img_vec_mat_reshape==0);
% 
% % Forcing the surrounding to black and symmetrize it
% for k = 1:3 %3 colour channels
%     temp = img_vec_mat_reshape(:,:,k);
%     temp(1:2,:) = 0;
%     temp(end-1:end,:) = 0;
%     temp(:,1:2) = 0;
%     temp(:,end-1:end) = 0;
%     temp_mask = ones(size(temp));
%     for i = 2:size(temp,1)-1
%         for j = 2:size(temp,2)-1
%             if temp(i,j)==0
%                 temp_mask(i-1:i+1,j-1:j+1) = 0;
%             end
%         end
%     end
%     
%     outMask(:,:,k) = temp_mask.*fliplr(temp_mask);
%     clear temp temp_mask
% end
% % Index of background (black)
% bck_common = find(outMask==0);

end


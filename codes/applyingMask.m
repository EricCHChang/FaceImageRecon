function outMat = applyingMask(inpMat,mask)
% Applying background mask on all images so that every image has the same
% background
% This is for the surface analysis
%Arguments:
%   inpMat - image cell array
%   mask - the background mask being applied on the images (note that this
%          is in 3 dimension (x pixels, y pixels, 3 colour channels) 
%Output:
%   outMat - image cell array after being masked with the background mask

% check if the mask is in 3 dimensions
if size(size(mask),2)==2
    mask=repmat(mask,1,1,3);
end

% apply mask on to each image
black = 0;
if iscell(inpMat)
    for i = 1:size(inpMat,1)
        img = inpMat{i,1};
        img(find(mask==black)) = 0; % apply mask (overlay black) on image
        outMat{i,1} = img;
    end
% else
%     for i=1:size(inpMat,4)
%         temp=outMat(:,:,:,i);
%         temp(find(temp==a))=a;
%         outMat(:,:,:,i)=temp;
%     end
end

end
        
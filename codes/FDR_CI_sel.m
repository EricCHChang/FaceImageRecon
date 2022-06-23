function [outMatFDR,outMatGen] = FDR_CI_sel(mat,q,minNumPix)
% the dimensions that have more than 'minNumPix' pixels are considered to
% be significant
% Inputs:
%       mat - p-values matrix (number of pixels x dimensions x images) for
%             leave-one-out analysis
%       q - false discovery rate
%       minNumPix - minimum number of pixels for a dimension to be
%                   considered significant

% Outputs:
%       OutMatFDR - significance of pixels after FDR correction (1:
%                   significant; 0: insignificant). number of pixels x
%                   dimensions x images
%       OutMatGen - number of stimulus x number of dimensions x number of
%                   colour channels; storing 0 or 1 indicating significant
%                   dimensions
%%
dim_max=size(mat,2);
outMatPix=NaN(size(mat)); 
chanSize=size(mat,1)/3; % number of colour channels
if ndims(mat)==3 % if the p-value matrix is from the leave-one-out analysis
    sel_res_FDR_bin=NaN(size(mat,3),dim_max,3);
    for ind_k=1:size(mat,3) % go through every face image
        for dim_k=1:dim_max % go through every dimension
            for j=1:3 % go through every colour channel
                rangChan=(j-1)*chanSize+1:j*chanSize; % indicies of pixels in the current channel
                [pID,~,~,~] = FDR_comp(squeeze(mat(rangChan,dim_k, ind_k)), q);
                if size(pID)>0
                    temp=zeros(size(rangChan));
                    temp=mat(rangChan,dim_k, ind_k)<=pID==1; % select significant pixels with p-values that are smaller than or equal to the FDR corrected threshold (pID)
                    sel_res_FDR_bin(ind_k, dim_k,j)=1;
                    if sum(temp)<minNumPix 
                        % if the number of significant pixels in 1 colour 
                        % channel, 1 dimension, 1 image is smaller than the 
                        % minimum required number of pixels
                        % this dimension in this image and in this colour
                        % channel is considered not significant
                        sel_res_FDR_bin(ind_k, dim_k,j)=0;
                        outMatPix(rangChan+(length(rangChan)*(j-1)),dim_k, ind_k)=0;
                    else
                        % if the number of significant pixels is more than
                        % the minimum required number of pixels, then
                        % consider it significant
                        % store the result of the FDR-corrected 
                        % significance for each pixel
                        outMatPix(rangChan+(length(rangChan)*(j-1)),dim_k, ind_k)=temp;
                    end
                else
                    % if none of the pixels survive FDR corrction, they're
                    % all insignificant
                    sel_res_FDR_bin(ind_k, dim_k,j)=0;
                end
            end
        end 
    end
end
outMatFDR=outMatPix;
outMatGen=sel_res_FDR_bin;
end


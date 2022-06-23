function [outMatGen] = FDR_CI_sel_shape(mat,q,minNumPix)
% the dimensions that have more than 'minNumPix' pixels are considered to
% be significant

% Outputs:
%       OutMatGen - number of stimulus x number of dimensions; 
%                   storing 0 or 1 indicating significant dimensions

%%
dim_max=size(mat,2); % number of dimensions
% outMatPix=NaN(size(mat));
% chanSize=size(mat,1)/3;
sel_res_FDR_bin=NaN(size(mat,3),dim_max); % number of images x number of dimensions
if ndims(mat)==3
%     sel_res_FDR_bin=NaN(size(mat,3),dim_max,3);
    for ind_k=1:size(mat,3)
        for dim_k=1:dim_max
%             for j=1:3
%                 rangChan=(j-1)*chanSize+1:j*chanSize;
%                 [pID,~,~,~] = FDR_comp(squeeze(mat(rangChan,dim_k, ind_k)), q);
                  temp=squeeze(mat(:,dim_k, ind_k));
                  [pID,~,~,~] = FDR_comp(temp, q);
                 
                if size(pID,1)>0 && sum(temp<=pID)>=minNumPix
%                     temp=zeros(size(rangChan));
%                     temp=mat(rangChan,dim_k, ind_k)<pID==1;
%                     sel_res_FDR_bin(ind_k, dim_k,j)=1;
                    sel_res_FDR_bin(ind_k, dim_k)=1;
%                     if sum(temp)<minNumPix
%                         sel_res_FDR_bin(ind_k, dim_k,j)=0;
%                         outMatPix(rangChan+(length(rangChan)*(j-1)),dim_k, ind_k)=0;
%                     else
%                         outMatPix(rangChan+(length(rangChan)*(j-1)),dim_k, ind_k)=temp;
%                     end
                else
%                     sel_res_FDR_bin(ind_k, dim_k,j)=0;
                    sel_res_FDR_bin(ind_k, dim_k)=0;
                end
%             end
        end 
    end
end
% outMatFDR=outMatPix;
outMatGen=sel_res_FDR_bin;
end


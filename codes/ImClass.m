function [pVal,CI]= ImClass(images,loadings,perm_n,leaveOut,bck_ind)
%%
% In line 32, the obsolete version of this function is run with temp=double(images{i}(:));
% After revision, this one is running with temp=images{i}(:);

%% based on CI_eval_node2.m
% Arguments:
%     images - cell array with images
%     loadings - results from parMDS function. Either 2D (identities X dims)...
%                 or 3D for leave one out (identites X dims x identities)
%     perm_n - number of desired permutations
%     leaveOut - 1 if you want to use a leave out procedure
%     bck_ind - indices for background pixels for one image and one color channel
% Outputs:
%     pVal - single matrix with p-values based on permutation test
%            number of pixels X number of dimensions for non-leave-one-out
%            analysis; number of pixels X number of dimensions x number of 
%            images for leave-one-out analysis
%     CI - a matrix storing classification images of every dimensions
%          (number of pixels x number of dimensions)

%% Prparation
tic
rng('shuffle')
temp=double(images{1});
if size(temp(:),1)~=max(bck_ind)
    error('Background index is not matching the images')
end
% ones_ind=find(temp(:));
ones_ind=1:max(bck_ind);
ones_ind=setdiff(ones_ind,bck_ind)';
% bck_ind=find(temp(:)==0); %index for background
im_mat=NaN(size(ones_ind,1),length(images));
for i=1:size(images,1)% converting image cell array to 2D array
    %temp=double(images{i}(:));
    temp=images{i}(:);
    im_mat(:,i)=temp(ones_ind);
end


sz=size(im_mat,1); % number of pixels (across 3 colour channels)
% CI_neut=NaN(size(im_mat,1),size(loadings,2));
CI=NaN(size(im_mat,1),size(loadings,2)); % number of pixels x number of dimensions
% CI_mat_neut=NaN(sz, perm_n);% matrix for permuations
CI_mat=NaN(sz, perm_n);

%% computing permutations
if ~leaveOut %for all identities at once
    Y_mat=zscore(loadings);
    for ROI_k=1:size(loadings,3) % go through each condition or brain region; if the loading matrix doesn't contain such information, this FOR loop does nothing (i.e., ROI_k = 1:1)
%         pval_CI_neut_mat=NaN(sz,size(loadings,2));
        pval_CI_mat=NaN(sz,size(loadings,2)); % number of pixels x dimensions
        for dim_k=1:size(loadings,2) %size(Y_mat, 2); % #dims for recon purposes
            parfor perm_k=1:perm_n+1
                if perm_k>1 
                    % coefficients in all but the first iterations are shuffled
                    ind_rand=randperm(length(Y_mat))';
                    Y_rnd=Y_mat(ind_rand,dim_k);
                else
                    % coefficients in the first iteration is not shuffled (i.e., actual/observed value)
                    Y_rnd=Y_mat(:,dim_k); 
                end
                
                ind_pos=find(Y_rnd(:,1)>0); % index of images with positive coordinate
                ind_neg=find(Y_rnd(:,1)<0); % index of images with negative coordinate
                
%                 im_mat_pos_neut=im_mat(:, ind_pos+size(loadings,1));
%                 im_mat_neg_neut=im_mat(:, ind_neg+size(loadings,1));
                im_mat_pos_hap=im_mat(:, ind_pos); % pixel x images with positive coordinate
                im_mat_neg_hap=im_mat(:, ind_neg);
                                
                Y_pos=Y_rnd(ind_pos,1); % positive coordinates 
                Y_neg=-Y_rnd(ind_neg,1); % negative coordinates (add a minus sign to make it positve; for calcualte weighted average face)
                                
                Y_pos_mat=repmat(Y_pos', [sz 1]); % column: coordinate of each image with positive coordinate
                Y_neg_mat=repmat(Y_neg', [sz 1]);
                
                %%% prots 
                % pixel values of an image are multipled by the
                % coordinate of that image; this is done for every image;
                % then, the multiplication results are summed across
                % images, separately for each pixel
%                 prot_pos_neut=sum(im_mat_pos_neut.*Y_pos_mat, 2);
%                 prot_neg_neut=sum(im_mat_neg_neut.*Y_neg_mat, 2);
                prot_pos=sum(im_mat_pos_hap.*Y_pos_mat, 2); 
                prot_neg=sum(im_mat_neg_hap.*Y_neg_mat, 2);
                
%                 % further divided by the sum of the coordinates
%                 % this step is optional because the classification image 
%                 % result doesn't change with or without this step
%                 prot_pos = sum(im_mat_pos_hap.*Y_pos_mat, 2) / sum(Y_pos); 
%                 prot_neg = sum(im_mat_neg_hap.*Y_neg_mat, 2) / sum(Y_neg);
                
                
                % put the classification image at 1 permutation iteration 
                % in a matrix (classification image was computed by
                % subracting the average face image of the negative group
                % form that of the positive group)
%                 CI_mat_neut(:,perm_k)=prot_pos_neut-prot_neg_neut;
                CI_mat(:,perm_k)=prot_pos-prot_neg;
                
        
            end
           
            CI(:,dim_k,ROI_k)=CI_mat(:,1); % the actual (observed) classificaiton image
%             CI_neut(:,dim_k,ROI_k)=CI_mat_neut(:,1);
%             pval_CI_neut_mat(:, dim_k,ROI_k)=comp_pval(CI_mat_neut);
            pval_CI_mat(:, dim_k, ROI_k)=comp_pval(CI_mat); % calcualte the significance of each pixel
            
            % to measure time
            display(['dimension ' num2str(dim_k)])
            toc

        end
        
        
        
        toc
    end
    
    pVal=squeeze(single(pval_CI_mat));
%     p_neut=squeeze(single(pval_CI_neut_mat));
    CI=squeeze(CI);
%     CI_neut=squeeze(CI_neut);

else % for leave one out analysis

    Y_mat=zscore(loadings);
    
    for ROI_k=1:size(loadings,4)
        
%         pval_CI_neut_mat=NaN(sz, size(loadings,2), size(loadings,1));
        pval_CI_mat=NaN(sz, size(loadings,2), size(loadings,1));
        
        for ind_k=1:size(loadings,3)
            rng('shuffle')
            Y_L1out=Y_mat(:,:,ind_k);            
            for dim_k=1:size(loadings,2)
%                 CI_mat_neut=NaN(sz, perm_n);
                CI_mat=NaN(sz, perm_n);
                parfor perm_k=1:perm_n+1
                    if perm_k>1
                        ind_rand=randperm(size(loadings,1))';
                        Y_rnd=Y_L1out(ind_rand,dim_k);
                    else
                        Y_rnd=Y_L1out(:,dim_k);
                    end
                    
                    ind_pos=find(Y_rnd(:,1)>0); % index of positive coordinate
                    ind_neg=find(Y_rnd(:,1)<0);
                    
%                     im_mat_pos_neut=im_mat(:, ind_pos+size(loadings,1));
%                     im_mat_neg_neut=im_mat(:, ind_neg+size(loadings,1));
                    im_mat_pos_hap=im_mat(:, ind_pos); % the image with positive coordinate
                    im_mat_neg_hap=im_mat(:, ind_neg);
                    
                    Y_pos=Y_rnd(ind_pos,1); % positive coordinate 
                    Y_neg=-Y_rnd(ind_neg,1);
                    
                    Y_pos_mat=repmat(Y_pos', [sz 1]);
                    Y_neg_mat=repmat(Y_neg', [sz 1]);
                    
                    %%%prots 
%                     prot_pos_neut=sum(im_mat_pos_neut.*Y_pos_mat, 2);
%                     prot_neg_neut=sum(im_mat_neg_neut.*Y_neg_mat, 2);
                    prot_pos=sum(im_mat_pos_hap.*Y_pos_mat, 2); % sum all positive images
                    prot_neg=sum(im_mat_neg_hap.*Y_neg_mat, 2);
                   
%                     CI_mat_neut(:,perm_k)=prot_pos_neut-prot_neg_neut;
                    CI_mat(:,perm_k)=prot_pos-prot_neg; % classfication image
                    
                    
                end
                
%                 pval_vect_neut=comp_pval(CI_mat_neut);
                pval_vect=comp_pval(CI_mat);
%                 pval_CI_neut_mat(:, dim_k, ind_k,ROI_k)=pval_vect_neut;
                pval_CI_mat(:, dim_k, ind_k,ROI_k)=pval_vect;
                
            end
            toc
            display(['identity ' num2str(ind_k)])
        end
        pVal=squeeze(single(pval_CI_mat));
%         p_neut=squeeze(single(pval_CI_neut_mat)); 
    end
    toc
end
% close(h)

end




%% computes signif based on permutation test
function pval_vect=comp_pval(CI_mat)
sz=size(CI_mat,1); % number of pixels (across 3 colour channels)
perm_n=size(CI_mat,2); % permutation times
tmp_sort=sort(abs(CI_mat), 2, 'descend');
tmp_act=abs(CI_mat(:,1));
rnk_vect=NaN(sz,1);
for val_k=1:sz
    rnk_vect(val_k, 1)=find(tmp_act(val_k,1)==tmp_sort(val_k,:), 1, 'last');
end
pval_vect=rnk_vect/perm_n; % the proportion of the values that are equal or greater than actual value 
pval_vect=pval_vect/2;%%1-tailed
end


function [p_happ,CI_happ]= ImClass_shape(shapes,loadings,perm_n,leaveOut)
%% based on CI_eval_node2.m
% Arguments:
%     shapes - matrix of shape information. Dimensions are: number of
%              images x number of fiducial points x coordinates (x and y)
%     loadings - results from patMDS function. Either 2D (identities X dims)...
%                 or 3D for leave one out (identites X dims x identities)
%     perm_n - number of desired permutations
%     leaveOut - 1 if you want to use a leave out procedure
% Outputs:
%     p_hap - single matrix with p-values based on permutation test
%                 (Pixels X dims X ids) for happy (actually for neutral
%                 faces in my study)

%% Prparation
tic
rng('shuffle')
shapes = permute(shapes,[2,3,1]);
%Dimension after reorganize: 1stDim: number of fiducial points; 2ndDim: coordinates (x&y); 3rdDim: number of face images 
shapes = reshape(shapes,size(shapes,1)*size(shapes,2),size(shapes,3));
%after the above step, the matrix is a 2D (x+y x number of face images),
%from 1st to 82nd rows are x coordinates and 83rd to 164th rows are y
%coordinates
sz = size(shapes); % sz(1): number of x+y coordinates; sz(2): number of images
p_happ=[];
% p_neut=[];
CI_happ=[];
% CI_neut=[];
% out_pos_neut=[];
% out_neg_neut=[];
out_pos_hap=[];
out_neg_hap=[];

% CI_neut=NaN(size(im_mat,1),size(loadings,2));
% CI_happ=NaN(size(im_mat,1),size(loadings,2));
% CI_mat_neut=NaN(sz, perm_n);% matrix for permuations
% CI_mat_happ=NaN(sz, perm_n);

%% computing permutations
if ~leaveOut %for all identities at once
    Y_mat=zscore(loadings);
    for ROI_k=1:size(loadings,3)
%         pval_CI_neut_mat=NaN(sz,size(loadings,2));
%         pval_CI_hap_mat=NaN(sz,size(loadings,2));
        for dim_k=1:size(loadings,2)%size(Y_mat, 2); % #dims for recon purposes
            for perm_k=1:perm_n+1
                if perm_k>1 % all but first layer are permuted.
                    ind_rand=randperm(length(Y_mat))';
                    Y_rnd=Y_mat(ind_rand,dim_k);
                else % first layer is not perumted (true)
                    Y_rnd=Y_mat(:,dim_k); 
                end
                
                ind_pos=find(Y_rnd(:,1)>0);
                ind_neg=find(Y_rnd(:,1)<0);
                
                
%                 im_mat_pos_neut=im_mat(:, ind_pos+size(loadings,1));
%                 im_mat_neg_neut=im_mat(:, ind_neg+size(loadings,1));
                im_mat_pos_hap=shapes(:, ind_pos); %coords X images
                im_mat_neg_hap=shapes(:, ind_neg); %coords X images
                
                
                Y_pos=Y_rnd(ind_pos,1); % positive coef
                Y_neg=-Y_rnd(ind_neg,1); % negative coef
                
                
%                 Y_pos_mat=repmat(Y_pos', [sz 1]);
%                 Y_neg_mat=repmat(Y_neg', [sz 1]);
                
                %%%prots - unscaled here
%                 prot_pos_neut=sum(im_mat_pos_neut.*Y_pos_mat, 2);
%                 prot_neg_neut=sum(im_mat_neg_neut.*Y_neg_mat, 2);
                prot_pos_hap=sum(im_mat_pos_hap*Y_pos, 2);
                prot_neg_hap=sum(im_mat_neg_hap*Y_neg, 2);
                
                if perm_k==1
%                     pr_pos_neut=prot_pos_neut;
%                     pr_neg_neut=prot_neg_neut;
                    pr_pos_hap=prot_pos_hap;
                    pr_neg_hap=prot_neg_hap;
                end
                
%                 CI_mat_neut(:,perm_k)=prot_pos_neut-prot_neg_neut;
                CI_mat_happ(:,perm_k)=prot_pos_hap-prot_neg_hap;
                
        
            end
           
%             out_pos_neut(:,dim_k,ROI_k)=pr_pos_neut;
%             out_neg_neut(:,dim_k,ROI_k)=pr_neg_neut;
            out_pos_hap(:,dim_k,ROI_k)=pr_pos_hap;
            out_neg_hap(:,dim_k,ROI_k)=pr_neg_hap;
            
            CI_happ(:,dim_k,ROI_k)=CI_mat_happ(:,1);
%             CI_neut(:,dim_k,ROI_k)=CI_mat_neut(:,1);
%             pval_CI_neut_mat(:, dim_k,ROI_k)=comp_pval(CI_mat_neut);
            pval_CI_hap_mat(:, dim_k,ROI_k)=comp_pval(CI_mat_happ);
            
            % to measure time
            display(['dimension ' num2str(dim_k)])
            toc

        end
        
       
        
        
        toc
    end
    
    p_happ=squeeze(single(pval_CI_hap_mat));
%     p_neut=squeeze(single(pval_CI_neut_mat));
    CI_happ=squeeze(CI_happ);
%     CI_neut=squeeze(CI_neut);

    toc
else % for leave one out analysis
    
    
    Y_mat=zscore(loadings);
    
    for ROI_k=1:size(loadings,4)
        
%         pval_CI_neut_mat=NaN(sz, size(loadings,2), size(loadings,1));
        pval_CI_hap_mat=NaN(sz(1), size(loadings,2), size(loadings,1)); % number of coordinates (x+y) x number of dimensions x number of images
        
        for ind_k=1:size(loadings,3)
            rng('shuffle')
            Y_L1out=Y_mat(:,:,ind_k);
            if size(loadings,1)==size(loadings,3)
                imgs=shapes;
            else
                error('need to check this code for memory design')
                indTrainHapp=1:size(loadings,1)-1;
%                 indTrainNeut=[1:size(loadings,1)-1]+size(im_mat,2)/2;
                indTestHapp=size(loadings,1)-1+ind_k;
%                 indTestNeut=size(loadings,1)-1+ind_k+size(im_mat,2)/2;
                imgs=shapes(:,[indTrainHapp indTestHapp]);
            end
            for dim_k=1:size(loadings,2)
%                 CI_mat_neut=NaN(sz, perm_n);
                CI_mat_happ=NaN(sz(1), perm_n);
                for perm_k=1:perm_n+1
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
                    im_mat_pos_hap=imgs(:, ind_pos); % the image with positive coordinate
                    im_mat_neg_hap=imgs(:, ind_neg);
                    
                    Y_pos=Y_rnd(ind_pos,1); % positive coordinate 
                    Y_neg=-Y_rnd(ind_neg,1);
                    
%                     Y_pos_mat=repmat(Y_pos', [sz(1) 1]);
%                     Y_neg_mat=repmat(Y_neg', [sz(1) 1]);
                    
                    %%%prots - unscaled here
%                     prot_pos_neut=sum(im_mat_pos_neut.*Y_pos_mat, 2);
%                     prot_neg_neut=sum(im_mat_neg_neut.*Y_neg_mat, 2);
                    prot_pos_hap=sum(im_mat_pos_hap*Y_pos, 2); % sum all positive images
                    prot_neg_hap=sum(im_mat_neg_hap*Y_neg, 2);
                   
%                     CI_mat_neut(:,perm_k)=prot_pos_neut-prot_neg_neut;
                    CI_mat_happ(:,perm_k)=prot_pos_hap-prot_neg_hap; % classfication image
                    
                    
                end
                
%                 pval_vect_neut=comp_pval(CI_mat_neut);
                pval_vect_hap=comp_pval(CI_mat_happ);
%                 pval_CI_neut_mat(:, dim_k, ind_k,ROI_k)=pval_vect_neut;
                pval_CI_hap_mat(:, dim_k, ind_k,ROI_k)=pval_vect_hap;
                
            end
            toc
            display(['identity ' num2str(ind_k)])
        end
        p_happ=squeeze(single(pval_CI_hap_mat));
%         p_neut=squeeze(single(pval_CI_neut_mat)); 
    end
    toc
end
% close(h)


end




%% computes signif based on permutation test
function pval_vect=comp_pval(CI_mat)
sz=size(CI_mat,1);
perm_n=size(CI_mat,2);
tmp_sort=sort(abs(CI_mat), 2, 'descend');
tmp_act=abs(CI_mat(:,1));
rnk_vect=NaN(sz,1);
for val_k=1:sz
    rnk_vect(val_k, 1)=find(tmp_act(val_k,1)==tmp_sort(val_k,:), 1, 'last');
end
pval_vect=rnk_vect/perm_n;
pval_vect=pval_vect/2;%%1-tailed
end
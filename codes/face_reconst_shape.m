function [recon_mat]=face_reconst_shape(CIsel_hap_bin,loads,shapes) %CIsel_neut_bin,
% Inputs
%       CIsel_hap_bin - which significant dimensions is included for
%                       reconstruction (number of images x number of
%                       dimensions)
%       loads - leave-one-out MDS loadings
%       shapes - matrix of all images' shape information

%% Converting my matrices to Adrian's format
% Converting image matrix into 2-D
dim_max=size(CIsel_hap_bin,2);
id_max=size(loads,1);
im_mat=[];
shapes=permute(shapes,[2,3,1]);
%Dimension after reorganize: 1stDim: number of fiducial points; 2ndDim: coordinates (x&y); 3rdDim: number of face images
shapes=reshape(shapes,size(shapes,1)*size(shapes,2),size(shapes,3));
%after the above step, the matrix is a 2D (x+y x number of face images),
%from 1st to 82nd rows are x coordinates and 83rd to 164th rows are y
%coordinates
[sz, im_n]=size(shapes);
% sz: number of x+y coordinates; im_n: number of non-targets
% for i=1:size(ims,1)% converting image cell array to 2D array
%     temp=double(ims{i}(:));
%     im_mat=[im_mat temp];
% end

% Converting bin matrices
% CIsel_neut_bin=reshape(CIsel_neut_bin,id_max*3,dim_max);
% CIsel_hap_bin=reshape(CIsel_hap_bin,id_max*3,dim_max);
% CIsel_hap_bin=squeeze(sum(CIsel_hap_bin,3)); % identity x dim
% CIsel_hap_bin=sum(CIsel_hap_bin,3); % this is not necessary for shape becasue there is no 3 colour channels
% [sz, im_n]=size(im_mat);
% sz_comp=sz/3;
% recon_mat=NaN(sz, im_n);
recon_mat=NaN(sz, im_n);

%% If no dimension is significant for sub X chan, setting first dimension as signficant
%%%find if no dims provide info (by perm test); select the 1st one then
% replace_ind=sum(CIsel_neut_bin, 2)==0; %60 x 1 x 3. looks for images X channels with no sig dimenstions
% CIsel_neut_bin(replace_ind, 1)=1; %180 x 20 x 3
replace_ind=sum(CIsel_hap_bin, 2)==0;
CIsel_hap_bin(replace_ind, 1)=1;


if size(loads,3)==size(loads,1)

    for ind_k=1:id_max
         ind_train=setdiff(1:id_max, ind_k);
         
%          im_orig_neut=im_mat(:,ind_k+id_max);
         im_orig_hap=shapes(:,ind_k);
%          im_orig_neut_train=im_mat(:,ind_train+id_max);
         im_orig_hap_train=shapes(:,ind_train);
         

         %%%find informative dimensions
%          CIsel_neut_curr=find(CIsel_neut_bin(ind_k,:));
         CIsel_hap_curr=find(CIsel_hap_bin(ind_k,:));
         
%          dim_sel_neut=size(CIsel_neut_curr, 2);
         dim_sel_hap=size(CIsel_hap_curr, 2); % number of sig. dimensions for this target image
         
         %%%select coefs for training images
         Y_curr=loads(:,:, ind_k);
         Y_L1out=Y_curr(ind_train,:);
         
%          Y_L1out_sel_neut=Y_L1out(:, CIsel_neut_curr); %choosing only sig loadings
         Y_L1out_sel_hap=Y_L1out(:, CIsel_hap_curr); %choosing only sig loadings
         
         %%%find the dist to origin for each training face for the purpose of generating 'origin' face 
         %%%use only diagnostic dims
%          dist_L1out_sel_neut=sqrt(sum(Y_L1out_sel_neut(:, 1:dim_sel_neut).^2, 2));
         dist_L1out_sel_hap=sqrt(sum(Y_L1out_sel_hap(:, 1:dim_sel_hap).^2, 2));
%          dist_L1out_sc=dist_L1out.^(-1);
%          dist_L1out_sc=dist_L1out_sc.*(1/sum(dist_L1out_sc, 1));
         %%%norm to unit for the purpose of generating 'origin' face
%          dist_L1out_sel_neut_sc=dist_L1out_sel_neut.*(1/sum(dist_L1out_sel_neut, 1));
         dist_L1out_sel_hap_sc=dist_L1out_sel_hap.*(1/sum(dist_L1out_sel_hap, 1));
         
% %          sum(dist_L1out_sel_neut_sc) %norm to 1 check
% %          sum(dist_L1out_sel_hap_sc)
% %          error
         
         %%%face 'origin' constr here
%          im_mn_neut=sum(im_orig_neut_train .* repmat(dist_L1out_sel_neut_sc', [sz 1]), 2);
         im_mn_hap=sum(im_orig_hap_train .* repmat(dist_L1out_sel_hap_sc', [sz(1) 1]), 2); %im_mn_hap=sum(im_orig_hap_train .* repmat(dist_L1out_sel_hap_sc', [sz 1]), 2);
% %          im_mn_neut=mean(cat(2, ims_pos_neut, ims_neg_neut), 2);
% %          im_mn_hap=mean(cat(2, ims_pos_hap, ims_neg_hap), 2);
         
         
         
% %          CI_mat_neut=NaN(sz1, sz2, sz3, dim_max);
% %          CI_mat_hap=NaN(sz1, sz2, sz3, dim_max);
%          CI_mat_neut=NaN(sz, dim_sel_neut);
         CI_mat_hap=NaN(sz(1), dim_sel_hap);
         
         %%%neut face constr
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%          for dim_k=1:dim_sel_neut
%                           
%              ind_pos=find(Y_L1out_sel_neut(:,dim_k)>0);
%              ind_neg=find(Y_L1out_sel_neut(:,dim_k)<0);
%              
%              ims_pos_neut=im_orig_neut_train(:, ind_pos);
%              ims_neg_neut=im_orig_neut_train(:, ind_neg);
%              
%              Y_pos=Y_L1out_sel_neut(ind_pos,dim_k);
%              Y_neg=-Y_L1out_sel_neut(ind_neg,dim_k);
%                                      
%              Y_pos_mat=repmat(Y_pos', [sz 1]);             
%              Y_neg_mat=repmat(Y_neg', [sz 1]);
%              
%              %%%prots - unscaled here
%              prot_pos_neut=sum(ims_pos_neut.*Y_pos_mat, 2);
%              prot_neg_neut=sum(ims_neg_neut.*Y_neg_mat, 2);
%              
%              CI_neut=prot_pos_neut-prot_neg_neut;
%              
%              
%              cf=Y_curr(ind_k, dim_k);
%              
%              CI_mat_neut(:, dim_k)=cf*CI_neut/2;
%              
%          end
         
                  
%          recon_im_neut=im_mn_neut+sum(CI_mat_neut, 2);
%          recon_mat(:,ind_k+id_max)=recon_im_neut;
         
        % check to assess recon appearance         
%         %          conv_im_RGB(im_orig_neut, sz_im, ones_ind, cform_lab2srgb)
%         %          conv_im_RGB(im_mn_neut, sz_im, ones_ind, cform_lab2srgb)
%         %          conv_im_RGB(recon_im_neut, sz_im, ones_ind, cform_lab2srgb)
%         %          error
        
        
         %%%hap face constr
         %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
         for dim_k=1:dim_sel_hap
                          
             ind_pos=find(Y_L1out_sel_hap(:,dim_k)>0); % find index of training images with positive coefficient
             ind_neg=find(Y_L1out_sel_hap(:,dim_k)<0);
             
             ims_pos_hap=im_orig_hap_train(:, ind_pos); % training images with positive coefficient
             ims_neg_hap=im_orig_hap_train(:, ind_neg);
             
             Y_pos=Y_L1out_sel_hap(ind_pos,dim_k); % positive coefficients
             Y_neg=-Y_L1out_sel_hap(ind_neg,dim_k);
                                     
             Y_pos_mat=repmat(Y_pos', [sz(1) 1]);             
             Y_neg_mat=repmat(Y_neg', [sz(1) 1]);
             
             %%%prots - unscaled here
             prot_pos_hap=sum(ims_pos_hap.*Y_pos_mat, 2); % prototype image from positvie sides (sum of images with positive coefficients)
             prot_neg_hap=sum(ims_neg_hap.*Y_neg_mat, 2);
             
             CI_hap=prot_pos_hap-prot_neg_hap; % classification image
             
             
             %cf=Y_curr(ind_k, dim_k);
             cf=Y_curr(ind_k, CIsel_hap_curr(dim_k)); % coefficients of to be reconstructed face
            
             
             CI_mat_hap(:, dim_k)=cf*CI_hap/2; % coefficients multiply classification image
             % it is divided by two because we subtract negative template
             % from positive template, the coefficient (distance) to the
             % origin is doubled.
             
         end
         
         recon_im_hap=im_mn_hap+sum(CI_mat_hap, 2); % origin face + sum of linear combinations of coefficents and classification images 
         recon_mat(:,ind_k)=recon_im_hap;
         
        %%%check to assess recon appearance         
% %          conv_im_RGB(im_orig_hap, sz_im, ones_ind, cform_lab2srgb)
% %          conv_im_RGB(im_mn_hap, sz_im, ones_ind, cform_lab2srgb)
% %          conv_im_RGB(recon_im_hap, sz_im, ones_ind, cform_lab2srgb)
% %          error
         
                    
    end
    %%%%%%%%%%%%%%%%%%% 
else
error('check this code for memory target design')
    for ind_k=1:size(loads,3)
        ind_train=1:size(loads,1)-1;
        
%         im_orig_neut=im_mat(:,ind_k+size(loads,1)-1+id_max);
        im_orig_hap=im_mat(:,ind_k+size(loads,1)-1);
%         im_orig_neut_train=im_mat(:,ind_train+id_max);
        im_orig_hap_train=im_mat(:,ind_train);
        
        
        %%%find informative dimensions
%         CIsel_neut_curr=find(CIsel_neut_bin(ind_k,:));
        CIsel_hap_curr=find(CIsel_hap_bin(ind_k,:));
        
%         dim_sel_neut=size(CIsel_neut_curr, 2);
        dim_sel_hap=size(CIsel_hap_curr, 2);
        
        %%%select coefs for training images
        Y_curr=loads(:,:, ind_k);
        Y_L1out=Y_curr(ind_train,:);
        
%         Y_L1out_sel_neut=Y_L1out(:, CIsel_neut_curr); %choosing only sig loadings
        Y_L1out_sel_hap=Y_L1out(:, CIsel_hap_curr); %choosing only sig loadings
        
        %%%find the dist to origin for each training face for the purpose of generating 'origin' face
        %%%use only diagnostic dims
%         dist_L1out_sel_neut=sqrt(sum(Y_L1out_sel_neut(:, 1:dim_sel_neut).^2, 2));
        dist_L1out_sel_hap=sqrt(sum(Y_L1out_sel_hap(:, 1:dim_sel_hap).^2, 2));
        %          dist_L1out_sc=dist_L1out.^(-1);
        %          dist_L1out_sc=dist_L1out_sc.*(1/sum(dist_L1out_sc, 1));
        %%%norm to unit for the purpose of generating 'origin' face
%         dist_L1out_sel_neut_sc=dist_L1out_sel_neut.*(1/sum(dist_L1out_sel_neut, 1));
        dist_L1out_sel_hap_sc=dist_L1out_sel_hap.*(1/sum(dist_L1out_sel_hap, 1));
        
        %          sum(dist_L1out_sel_neut_sc) %norm to 1 check
        %          sum(dist_L1out_sel_hap_sc)
        %          error
        
        %%%face 'origin' constr here
%         im_mn_neut=sum(im_orig_neut_train .* repmat(dist_L1out_sel_neut_sc', [sz 1]), 2);
        im_mn_hap=sum(im_orig_hap_train .* repmat(dist_L1out_sel_hap_sc', [sz(1) 1]), 2);
        %          im_mn_neut=mean(cat(2, ims_pos_neut, ims_neg_neut), 2);
        %          im_mn_hap=mean(cat(2, ims_pos_hap, ims_neg_hap), 2);
        
        
        
        %          CI_mat_neut=NaN(sz1, sz2, sz3, dim_max);
        %          CI_mat_hap=NaN(sz1, sz2, sz3, dim_max);
%         CI_mat_neut=NaN(sz, dim_sel_neut);
        CI_mat_hap=NaN(sz, dim_sel_hap);
        
        %%%neut face constr
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%         for dim_k=1:dim_sel_neut
%             
%             ind_pos=find(Y_L1out_sel_neut(:,dim_k)>0);
%             ind_neg=find(Y_L1out_sel_neut(:,dim_k)<0);
%             
%             ims_pos_neut=im_orig_neut_train(:, ind_pos);
%             ims_neg_neut=im_orig_neut_train(:, ind_neg);
%             
%             Y_pos=Y_L1out_sel_neut(ind_pos,dim_k);
%             Y_neg=-Y_L1out_sel_neut(ind_neg,dim_k);
%             
%             Y_pos_mat=repmat(Y_pos', [sz 1]);
%             Y_neg_mat=repmat(Y_neg', [sz 1]);
%             
%             %%%prots - unscaled here
%             prot_pos_neut=sum(ims_pos_neut.*Y_pos_mat, 2);
%             prot_neg_neut=sum(ims_neg_neut.*Y_neg_mat, 2);
%             
%             CI_neut=prot_pos_neut-prot_neg_neut;
%             
%             
%             cf=Y_curr(ind_k, CIsel_neut_curr(dim_k));
%             
%             CI_mat_neut(:, dim_k)=cf*CI_neut/2;
%             
%         end
%         
%         
%         recon_im_neut=im_mn_neut+sum(CI_mat_neut, 2);
%         recon_mat(:,ind_k+size(loads,3))=recon_im_neut;
%         
        % check to assess recon appearance
        %          conv_im_RGB(im_orig_neut, sz_im, ones_ind, cform_lab2srgb)
        %          conv_im_RGB(im_mn_neut, sz_im, ones_ind, cform_lab2srgb)
        %          conv_im_RGB(recon_im_neut, sz_im, ones_ind, cform_lab2srgb)
        %          error
        
        
        %%%hap face constr
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        for dim_k=1:dim_sel_hap
            
            ind_pos=find(Y_L1out_sel_hap(:,dim_k)>0);
            ind_neg=find(Y_L1out_sel_hap(:,dim_k)<0);
            
            ims_pos_hap=im_orig_hap_train(:, ind_pos);
            ims_neg_hap=im_orig_hap_train(:, ind_neg);
            
            Y_pos=Y_L1out_sel_hap(ind_pos,dim_k);
            Y_neg=-Y_L1out_sel_hap(ind_neg,dim_k);
            
            Y_pos_mat=repmat(Y_pos', [sz 1]);
            Y_neg_mat=repmat(Y_neg', [sz 1]);
            
            %%%prots - unscaled here
            prot_pos_hap=sum(ims_pos_hap.*Y_pos_mat, 2);
            prot_neg_hap=sum(ims_neg_hap.*Y_neg_mat, 2);
            
            CI_hap=prot_pos_hap-prot_neg_hap;
            
            
            cf=Y_curr(ind_k, CIsel_hap_curr(dim_k));
            
            CI_mat_hap(:, dim_k)=cf*CI_hap/2;
            
        end
        
        recon_im_hap=im_mn_hap+sum(CI_mat_hap, 2);
        recon_mat(:,ind_k)=recon_im_hap;
        
        %%%check to assess recon appearance
        %          conv_im_RGB(im_orig_hap, sz_im, ones_ind, cform_lab2srgb)
        %          conv_im_RGB(im_mn_hap, sz_im, ones_ind, cform_lab2srgb)
        %          conv_im_RGB(recon_im_hap, sz_im, ones_ind, cform_lab2srgb)
        %          error
    
    end

end    

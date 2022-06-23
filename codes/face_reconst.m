function [recon_mat,diagnostic]=face_reconst(CIsel_hap_bin,loads,ims,coef) %CIsel_neut_bin,
%% 
%This function has been revised to add a new input argument: coef
%This allows you to decide whether you want to match the contrast between
%the original images and reconstructions when running this script (face
%reconstruction)

% Inputs
%       CIsel_hap_bin - which significant dimensions is included for
%                       reconstruction
%       loads - leave-one-out MDS loadings
%       ims - image cell array of all images
%       coef - whether to match contrast between the reconstruction and the
%              original image 
% Outputs:
%       recon_mat - number of pixels (including background pixels) x images
%       diagnostic - a structure storing information of contrast (for
%                    matching contrast in reconstructed images)

%% Converting my matrices to Adrian's format
% Converting image matrix into 2-D

if nargin<4 % no coef, there is inflation (matching contrast)
    blow=1;
elseif coef==1 % no inflation (not matching contrast)
    blow=0;
else % do inflation with value of coef
    blow=2;
end
coefs=[];
dim_max=size(CIsel_hap_bin,2);
id_max=size(loads,1);
im_mat=[];
for i=1:size(ims,1)% converting image cell array to 2D array
    %temp=double(ims{i}(:));
    temp=ims{i}(:);
    im_mat=[im_mat temp];
end

true_sz=size(im_mat,1);
bck=find(mean(im_mat,2)==0);
im_mat(bck,:)=[]; % removing background (removing pixel with value of zero)
av_im=mean(im_mat,2);
diagnostic.stims.av_conts=std(reshape(av_im,size(av_im,1)/3,3));
diagnostic.stims.conts_minus_av=squeeze(std(reshape(im_mat-repmat(av_im,1,size(im_mat,2)),size(im_mat,1)/3,3,size(im_mat,2))))';

% Arranging selection matrix
% Converting bin matrices
% CIsel_neut_bin=reshape(CIsel_neut_bin,id_max*3,dim_max);
% CIsel_hap_bin=reshape(CIsel_hap_bin,id_max*3,dim_max);
CIsel_hap_bin=squeeze(sum(CIsel_hap_bin,3)); % identity x dim
[sz, im_n]=size(im_mat); 
sz_comp=sz/3; % number of pixels of a image in each colour channel
recon_mat=NaN(true_sz, im_n); % Preparing reconstruction image matrix
diagnostic.stims.conts(:,:)=squeeze(std(reshape(im_mat,sz_comp,3,im_n)));
diagnostic.stims.mean(:,:)=squeeze(mean(reshape(im_mat,sz_comp,3,im_n)));

%% If no dimension is significant for sub X chan, setting first dimension as signficant
%%%find if no dims provide info (by perm test); select the 1st one then
% replace_ind=sum(CIsel_neut_bin, 2)==0; %60 x 1 x 3. looks for images X channels with no sig dimenstions
% CIsel_neut_bin(replace_ind, 1)=1; %180 x 20 x 3
replace_ind=sum(CIsel_hap_bin, 2)==0; % sum significance across colour channels and check whether there is any images that has zero significant channel
CIsel_hap_bin(replace_ind, 1)=1; % for images that have zero significance colour channel, use only the 1st dimension in reconstruction

for ind_k=1:id_max % go through every image
        
         ind_train=setdiff(1:id_max, ind_k);
         
%          im_orig_neut=im_mat(:,ind_k+id_max);
         im_orig_hap=im_mat(:,ind_k);
%          im_orig_neut_train=im_mat(:,ind_train+id_max);
         im_orig_hap_train=im_mat(:,ind_train);
         %targ_cont_hap=squeeze(mean(std(reshape(im_orig_hap_train,sz_comp,3,im_n-1)),3));
         targ_cont_hap=std(reshape(im_orig_hap,sz_comp,3)); %select the same image of reconstrction for matching contrast
         diagnostic.targ(ind_k,:)=targ_cont_hap;
         
         
         %%%find informative dimensions
%          CIsel_neut_curr=find(CIsel_neut_bin(ind_k,:));
         CIsel_hap_curr=find(CIsel_hap_bin(ind_k,:)); % significant dimension for the current image
         
%          dim_sel_neut=size(CIsel_neut_curr, 2);
         dim_sel_hap=size(CIsel_hap_curr, 2); % number of significant dimensions for the current image
         
         %%%select coefs for training images
         Y_curr=loads(:,:, ind_k); % projected coordinates of the current image (i.e., the left-out one) and other images
         Y_L1out=Y_curr(ind_train,:); % coordinates of other images in the face space constructed based on the n-1 remained faces
         
%          Y_L1out_sel_neut=Y_L1out(:, CIsel_neut_curr); %choosing only sig loadings
         Y_L1out_sel_hap=Y_L1out(:, CIsel_hap_curr); %choosing only sig loadings
         
         %%%find the dist to origin for each training face for the purpose of generating 'origin' face 
         %%%use only diagnostic dims
%          dist_L1out_sel_neut=sqrt(sum(Y_L1out_sel_neut(:, 1:dim_sel_neut).^2, 2));
         dist_L1out_sel_hap=sqrt(sum(Y_L1out_sel_hap(:, 1:dim_sel_hap).^2, 2)); %(:, 1:dim_sel_hap) can be deleted, it won't affect the output
%          dist_L1out_sc=dist_L1out.^(-1);
%          dist_L1out_sc=dist_L1out_sc.*(1/sum(dist_L1out_sc, 1));
         %%%norm to unit for the purpose of generating 'origin' face
%          dist_L1out_sel_neut_sc=dist_L1out_sel_neut.*(1/sum(dist_L1out_sel_neut, 1));
         dist_L1out_sel_hap_sc=dist_L1out_sel_hap.*(1/sum(dist_L1out_sel_hap, 1));
         
% %          sum(dist_L1out_sel_neut_sc) %norm to 1 check
% %          sum(dist_L1out_sel_hap_sc)
% %          error
         
         %%%face 'origin' constr here
         % pixel values of each of the other images multiplied by their own
         % distance to the origin (scaled)
         % then, sum across all images 
         % this step creates an weighted average face image, averaged
         % across all images other than the left-out image
%          im_mn_neut=sum(im_orig_neut_train .* repmat(dist_L1out_sel_neut_sc', [sz 1]), 2);
         im_mn_hap=sum(im_orig_hap_train .* repmat(dist_L1out_sel_hap_sc', [sz 1]), 2); 
% %          im_mn_neut=mean(cat(2, ims_pos_neut, ims_neg_neut), 2);
% %          im_mn_hap=mean(cat(2, ims_pos_hap, ims_neg_hap), 2);
         
         
         
% %          CI_mat_neut=NaN(sz1, sz2, sz3, dim_max);
% %          CI_mat_hap=NaN(sz1, sz2, sz3, dim_max);
%          CI_mat_neut=NaN(sz, dim_sel_neut);
         CI_mat_hap=NaN(sz, dim_sel_hap);
         
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
         for dim_k=1:dim_sel_hap % go through each significant dimension
                          
             ind_pos=find(Y_L1out_sel_hap(:,dim_k)>0); % find index of training images with positive coefficient
             ind_neg=find(Y_L1out_sel_hap(:,dim_k)<0);
             
             ims_pos_hap=im_orig_hap_train(:, ind_pos); % training images with positive coefficient
             ims_neg_hap=im_orig_hap_train(:, ind_neg);
             
             Y_pos=Y_L1out_sel_hap(ind_pos,dim_k); % positive coefficients
             Y_neg=-Y_L1out_sel_hap(ind_neg,dim_k);
                                     
             Y_pos_mat=repmat(Y_pos', [sz 1]);             
             Y_neg_mat=repmat(Y_neg', [sz 1]);
             
             %%%prots - unscaled here
             prot_pos_hap=sum(ims_pos_hap.*Y_pos_mat, 2); % prototype image from positvie sides (sum of images with positive coefficients)
             prot_neg_hap=sum(ims_neg_hap.*Y_neg_mat, 2);
             
             CI_hap=prot_pos_hap-prot_neg_hap; % classification image
             
             
             %cf=Y_curr(ind_k, dim_k);
             cf=Y_curr(ind_k, CIsel_hap_curr(dim_k)); % coefficients of to be reconstructed face in the current selected dimension
            
             
             CI_mat_hap(:, dim_k)=cf*CI_hap/2; % coefficients multiply classification image
             % it is divided by two because we subtract negative template
             % from positive template, the coefficient (distance) to the
             % origin is doubled.
             
         end
         
         avrg=reshape(im_mn_hap,size(im_mn_hap,1)/3,3); % the origin face
         cnt=reshape(sum(CI_mat_hap, 2),size(CI_mat_hap,1)/3,3); % reconstruction before matching contrast
         diagnostic.input.conts.avrg(ind_k,:)=std(avrg);
         diagnostic.input.mean.avrg(ind_k,:)=mean(avrg);
         diagnostic.input.conts.centered(ind_k,:)=std(cnt);
         diagnostic.input.mean.centered(ind_k,:)=mean(cnt);
         if blow == 1
             
%              display(['image number: ' num2str(ind_k)])
             
             [coef(1),all(:,1)]=compContCoeff(avrg(:,1),cnt(:,1),targ_cont_hap(1));
             [coef(2),all(:,2)]=compContCoeff(avrg(:,2),cnt(:,2),targ_cont_hap(2));
             [coef(3),all(:,3)]=compContCoeff(avrg(:,3),cnt(:,3),targ_cont_hap(3));
             diagnostic.coefs(:,1,ind_k)=all(:,1);
             diagnostic.coefs(:,2,ind_k)=all(:,2);
             diagnostic.coefs(:,3,ind_k)=all(:,3);
         elseif blow == 0
             coef=1;
         elseif blow == 2
         else
             error('Target contrast is not specified correctly')
         end
         
         diagnostic.addpart.conts(ind_k,1)=std(coef(1)*cnt(:,1));
         diagnostic.addpart.conts(ind_k,2)=std(coef(2)*cnt(:,2));
         diagnostic.addpart.conts(ind_k,3)=std(coef(3)*cnt(:,3));
         
         recon_im_hapL=avrg(:,1)+coef(1)*cnt(:,1);
         recon_im_hapa=avrg(:,2)+coef(2)*cnt(:,2);
         recon_im_hapb=avrg(:,3)+coef(3)*cnt(:,3);
         
         diagnostic.recon.conts(1,ind_k)=std(recon_im_hapL);
         diagnostic.recon.mean(1,ind_k)=mean(recon_im_hapL);
         diagnostic.recon.conts(2,ind_k)=std(recon_im_hapa);
         diagnostic.recon.mean(2,ind_k)=mean(recon_im_hapa);
         diagnostic.recon.conts(3,ind_k)=std(recon_im_hapb);
         diagnostic.recon.mean(3,ind_k)=mean(recon_im_hapb);
         
         recon_im_hap=ones(max(bck),1);
         recon_im_hap(bck)=0;
         recon_im_hap(recon_im_hap==1)=[recon_im_hapL;recon_im_hapa;recon_im_hapb];
%          recon_im_hap=im_mn_hap+sum(CI_mat_hap, 2);
         recon_mat(:,ind_k)=recon_im_hap;
         
        %%%check to assess recon appearance         
% %          conv_im_RGB(im_orig_hap, sz_im, ones_ind, cform_lab2srgb)
% %          conv_im_RGB(im_mn_hap, sz_im, ones_ind, cform_lab2srgb)
% %          conv_im_RGB(recon_im_hap, sz_im, ones_ind, cform_lab2srgb)
% %          error
         
                    
end
    
    

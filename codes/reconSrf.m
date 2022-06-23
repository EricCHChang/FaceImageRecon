function data = reconSrf(group,subID,parameter,saveFileDir,workDir,conf,imgFile,bck_common,objAccPerm)
% Perception-based surface reconstruction of facial identities used in the 
% DG Lesion study (Patient B.L.)
% Note: the similarity rating task and the reconstruction procedure are the 
% same as that in the aging study; here I just wrote a new script which has
% improved codes

% Input arguments:
%       group - subject from the patient('p') or healthy control group('c')
%       subID - subject ID
%       parameter - maximum number of dimensions, permutation times for
%                   creating classification images, criterion of minimum 
%                   number of significant pixels, and criterion for FDR 
%                   correction
%       saveFileDir - directory for saving the output file of this function
%       workDir - where this script is on the computer
%       conf - the confusability matrix of the perception-based task
%       imgFile - image cell arry of the original images (being masked by
%                 common background)
%       bck_common - maximum common background (indices, black) across all original images
%       objAccPerm - whether running permutation test for objective
%                    accuracy test
%
% Output:
%       data - a structure array containing all outputs

% Writtern by C.-H. Eric Chang, Feb 2020
% Updated on March, 2021 - add permutation test for each L*a*b* channel

%% Inputs
data.group = group;
data.subID = subID;
data.conf = conf;
data.imgFile = imgFile;
data.bck_common = bck_common;
data.objAccPerm = objAccPerm;

data.maxDim = parameter.maxDim; % number of MDS to retain
data.permutN = parameter.permutN; % number of permuations to perfrom
data.minNumPix = parameter.minNumPix; % minimum number of pixels to include as a signficiant dimension
data.q = parameter.q; % criterion (Q) for the FDR correction
data.accPermN = parameter.accPermN; % number of permutations for the significance of objective accuracy

data.imNum = size(conf,1); % nubmer of images
cd(workDir);

%% Converting files to LAB space
% Optional: Testing rgb to LAB conversion and back (set test to 1)
[data.labout] = convLab(data.imgFile,0); %converting to LAB space. 0-no testing

%% Computing MDS
% computes visualization all identity included matrix loadings57 (ids X MDS dims) 
% and leave-one-out matrix for permuations loadings56 (ids X ids X MDS dims) 
% Important: loadings56 contains a procurstian alignment weights for each identity N
% at n X MDS_dims X n
% the last input can be replaced by [1:57] if only names have to be presented
[data.loadAll,data.loadLeaveOut,~,data.vari_expl,data.vari_exp_cum] = patMDS(data.conf,data.maxDim,0.0001);
%[data.loadAll,data.loadLeaveOut,~,data.vari_expl,data.vari_exp_cum]=patMDS(data.conf,data.maxDim,0.0001,data.imgFile); 
% [~,~,~,~,~]=patMDS(data.conf,data.maxDim,0.0001,[1:25]);

%% Computing the prototypes for desired dimensions
% Computing the prototypes of specific dimensions
% Postive prototype is the mean face averaged across faces with positive
% z-transformed coefficients in one dimension
% Negative protopye is the mean face averaged across faces with negative
% z-transformed coefficients in one dimension

data.desiredDim = [1,2,3];
data.normalization = 1;
vis = 0;
run = 1; 

if run
    [data.proto_pos,data.proto_neg]=protoPresent(data.loadAll,data.imgFile,data.desiredDim,data.normalization,vis);
    proto_pos = data.proto_pos;
    proto_neg = data.proto_neg;
    normalization = data.normalization;
    outFile=[saveFileDir 'proto_' group num2str(subID) '_' date];
    save(outFile,'proto_pos','proto_neg','normalization');
else
%     load([saveFileDir '/proto_s01_18-Aug-2017.mat'])
%     for i=1:length(data.desiredDim)
%         figure
%         suptitle(['Prototype: Dimension ' num2str(data.desiredDim(i))]);
%         subplot (1,2,1)
%         imshow(uint8(data.proto_neg))
%         
%         subplot (1,2,2)
%         imshow(uint8(data.proto_pos))
%     end
end

%% Computing prototypes for visualization purposes
% Computing classification image based on LAB-converted stimuli and z-scored loadings
% Function ImClass first checks the dimensionality of the loading matrix.
% First it converts loadings to z scores.
% If the dimension number of the loading matrix is 2 it performs one 
% permutation test based on permutN number. Practically, for each pixel all
% permutation results are ranked and a percentile of a pixel value based on 
% the correct loading matrix is evaluted in a one-tailed manner.
% If the dimension number is 3 it itireates through all identities 
% and computes ranking similarly to described above. The output is p values
% matrices, classification images. bck is a vector containing background
% pixels and their position. 

%load('bck_common.mat') % load the indices of background (black) in the image
run = 1; 
if run
    [data.p_All,data.CI]=ImClass(data.labout,data.loadAll,data.permutN,0,data.bck_common); %p_happ60,CI_happy,
    outFile = [saveFileDir 'CI_All_' group num2str(subID) '_perm' num2str(data.permutN) '_' date];
    p_All = data.p_All;
    CI = data.CI;
    save(outFile,'p_All','CI'); %'p_happ57','CI_happy'
else
    %load([saveFileDir '/pval_All_s01_perm100_18-Oct-2017.mat'])
    %load([saveFileDir '/pval_All_s01_perm100_18-Aug-2017.mat'])
end

%% visualization of significant pixels for single permutation test
vis = 0;
if vis
    [data.pIdsN,data.CImat] = disp_CIs(data.CI,data.p_All,data.q,data.minNumPix,data.bck_common,data.imgFile,'Neutral',data.maxDim); 
%     [pIdsH,~]=disp_CIs(CI_happy,p_happ60,data.q,data.minNumPix,bck,ims_new,'Happy',20);
end

%% FDR corrected iterations with leave-one-out
run = 1;
if run
    [data.p_LOut,~]=ImClass(data.labout,data.loadLeaveOut,data.permutN,1,data.bck_common); %p_happ59
    outFile=[saveFileDir 'CI_LOut_' group num2str(subID) '_perm' num2str(data.permutN) '_' date];
    p_LOut = data.p_LOut;
    save(outFile,'p_LOut'); %'p_happ59'
else
    %load([saveFileDir '/s01_perm100_18-Oct-2017.mat'])
    %load([saveFileDir '/s01_perm100_18-Aug-2017.mat'])
end
[~,data.outMatGen] = FDR_CI_sel(data.p_LOut,data.q,data.minNumPix);

%% vizualization of significant dimensions
vis = 0;
if vis
    fig=figure;
    set(fig, 'Position', [100, 100, 800, 695]);
    for i=1:3
    subplot(1,3,i)
    imagesc(squeeze(data.outMatGen(:,:,i)));
    title(['Neutral channel ' num2str(i)]);
%     subplot(2,3,i+3)
%     imagesc(squeeze(outMatGen_happ(:,:,i)));
%     title(['Happy channel ' num2str(i)]);
    end
end

%% chosing significant pixels for iterations
% In this part we perform following operations:
% 1.    Find significant dimensions based on permutations from ImClass.m
%       If no significant dimension is found the first dimension is used.
% 2.    Using sig dimension find distances of all training faces to origin.
% 3.    Norm all distances, so that sum of all distances is 1.
% 4.    Construct origin face.
% 5.    For each sig dim: Compute positive and negative averaged faces.
% 6.    For each sig dim: Compute CI by subtracting negative from positive.
% 7.    Multiply reconstruction by it's coordinates on the fitted face
%       space
% 8.    Divide CI by 2 to account for traversing the distance twice.
% 9.    Add origin  
% 10.    #5-9 are repeated for both emotions.
[data.recon_mat,data.diagnostic] = face_reconst(data.outMatGen,data.loadLeaveOut,data.labout); %outMatGen_happ,

%% arranging back into row x column X color X dim array
data.recon_mat_sq = reshape(data.recon_mat,size(imgFile{1},1),size(imgFile{1},2),size(imgFile{1},3),size(imgFile,1));

%% converting reconstructed faces into RGB
for i=1:size(data.recon_mat_sq,4)
%     recon_mat_sq_rgb(:,:,:,i) = Lab2RGB(data.recon_mat_sq(:,:,:,i));
    %recon_mat_sq_rgb(:,:,:,i) = lab2rgb(data.recon_mat_sq(:,:,:,i));
    data.recon_mat_sq_rgb(:,:,:,i) = lab2rgb(data.recon_mat_sq(:,:,:,i),'OutputType','uint8');
end
%data.recon_mat_sq_rgb=uint8(data.recon_mat_sq_rgb);

%% visualization of the image
vis = 0;
imageN = 34; % visualize which image
if vis
    fig=figure;
    set(fig, 'Position', [100, 100, 800, 500]);
    subplot(1,2,1)
    imagesc(squeeze(data.recon_mat_sq_rgb(:,:,:,imageN)));
    title(['Reconstructed identity number ' num2str(imageN)])
    subplot(1,2,2)
    imagesc(imgFile{imageN});
    title(['Original identity number ' num2str(imageN)])
end

%% Save the reconstructed 57 unfamiliar faces (recon)
saveImg = 0;
for i = 1:data.imNum
    data.ims_reconTp_surface{i,1} = data.recon_mat_sq_rgb(:,:,:,i);
end
if saveImg 
    ims_reconTp_surface = data.ims_reconTp_surface;
    reconOut=fullfile(saveFileDir,['ims_reconTp_surface_' group num2str(subID) '_id' num2str(identity) '_' date '.mat']);
    save(reconOut,'ims_reconTp_surface')
end

%% running objective test 
% reconstructed image compared with pairs (original image and all different
% images) based on Euclidean distance. If the original image is closer a
% score of 1 is awarded, if further a score of 0 is awarded. Then all the
% scores are averaged per image, and the grand average of all reconstructed
% faces is compared against 0.5 using t-test.
if objAccPerm==0
    [~,data.p_val,data.aver_im,data.aver_all]=obj_test(data.recon_mat_sq(:,:,:,1:data.imNum),data.labout(1:data.imNum),0.05,1);
    %obj_test_surface is actually equal to obj_test
    
    % Compute the reconstruction accuracy for each colour channel
    % separately
    [~,p_val_L,aver_im_L,aver_all_L]=obj_test(data.recon_mat_sq(:,:,:,1:data.imNum),data.labout(1:data.imNum),0.05,'L');
    [~,p_val_A,aver_im_A,aver_all_A]=obj_test(data.recon_mat_sq(:,:,:,1:data.imNum),data.labout(1:data.imNum),0.05,'A');
    [~,p_val_B,aver_im_B,aver_all_B]=obj_test(data.recon_mat_sq(:,:,:,1:data.imNum),data.labout(1:data.imNum),0.05,'B');
    data.p_val_chan = [p_val_L p_val_A p_val_B]; %1x3 matrix
    data.aver_im_chan = [aver_im_L aver_im_A aver_im_B]; %57x3 matrix
    data.aver_all_chan = [aver_all_L aver_all_A aver_all_B]; %1x3 matrix
    
else
    [~,data.p_val,data.aver_im,data.aver_all]=obj_test(data.recon_mat_sq(:,:,:,1:data.imNum),data.labout(1:data.imNum),0.05,1);
    
    % Compute the reconstruction accuracy for each colour channel
    % separately
    [~,p_val_L,aver_im_L,aver_all_L]=obj_test(data.recon_mat_sq(:,:,:,1:data.imNum),data.labout(1:data.imNum),0.05,'L');
    [~,p_val_A,aver_im_A,aver_all_A]=obj_test(data.recon_mat_sq(:,:,:,1:data.imNum),data.labout(1:data.imNum),0.05,'A');
    [~,p_val_B,aver_im_B,aver_all_B]=obj_test(data.recon_mat_sq(:,:,:,1:data.imNum),data.labout(1:data.imNum),0.05,'B');
    data.p_val_chan = [p_val_L p_val_A p_val_B]; %1x3 matrix
    data.aver_im_chan = [aver_im_L aver_im_A aver_im_B]; %57x3 matrix
    data.aver_all_chan = [aver_all_L aver_all_A aver_all_B]; %1x3 matrix
    
    loadLeaveOut = data.loadLeaveOut;
    maxDim = data.maxDim;
    imNum = data.imNum;
    outMatGen = data.outMatGen;
    labout = data.labout;
    disp('permutation test for reconstruction accuracy started')
    parfor k = 1:data.accPermN
        loadMatPerm = loadLeaveOut;
        for i = 1:maxDim
            for j = 1:imNum
                temp = loadMatPerm(:,i,j);
                temp(j) = []; % leave the target out
                temp(randperm(imNum-1)) = temp; 
                loadMatPerm(setdiff(1:imNum,j),i,j) = temp; % randomly shuffled the coefficients of remained non-targets
            end
        end
        [recon_matPerm,~] = face_reconst(outMatGen,loadMatPerm,labout); % re-do reconstruction based on randomly shuffled MDS loadings
        recon_matPerm_sq = reshape(recon_matPerm,size(imgFile{1},1),size(imgFile{1},2),size(imgFile{1},3),size(imgFile,1));
        
        [~,~,aver_im_Perm(:,k),aver_all_Perm(1,k)] = obj_test(recon_matPerm_sq(:,:,:,1:imNum),labout(1:imNum),0.05,1); 
        % separate for channels
        [~,~,aver_im_L_Perm(:,k),aver_all_L_Perm(1,k)] = obj_test(recon_matPerm_sq(:,:,:,1:imNum),labout(1:imNum),0.05,'L');
        [~,~,aver_im_A_Perm(:,k),aver_all_A_Perm(1,k)] = obj_test(recon_matPerm_sq(:,:,:,1:imNum),labout(1:imNum),0.05,'A');
        [~,~,aver_im_B_Perm(:,k),aver_all_B_Perm(1,k)] = obj_test(recon_matPerm_sq(:,:,:,1:imNum),labout(1:imNum),0.05,'B');
    end
    disp('permutation test for reconstruction accuracy done')
    data.aver_im_Perm = aver_im_Perm;
    data.aver_all_Perm = aver_all_Perm;
    data.aver_im_Perm_chan = cat(3,aver_im_L_Perm,aver_im_A_Perm,aver_im_B_Perm); % nStim x nPerm x channel
    data.aver_all_Perm_chan = cat(3,aver_all_L_Perm,aver_all_A_Perm,aver_all_B_Perm); % 1 x nPerm x channel
    
    % significance of permutation test for reconstruction accuracy
    tmp_sort = sort(abs(data.aver_all_Perm), 'descend');
    tmp_act = abs(data.aver_all);
    data.rnk = find(tmp_sort>=tmp_act, 1, 'last'); % how many permuted accuracy is greater or equal to the actual accuracy
    if isempty(data.rnk)
        data.rnk = 0;
    end
    data.p_val_Perm = data.rnk/data.accPermN; % the proportion of the values that are equal or greater than actual value
    data.p_val_Perm_onetail = data.p_val_Perm/2; % 1-tailed; 
    data.alphaLevel = 0.05;
    data.CI_perm(1) = prctile(tmp_sort,100*data.alphaLevel/2); %CI lower bound
    data.CI_perm(2) = prctile(tmp_sort,100-100*data.alphaLevel/2); % CI upper bound
    
    % significance of permutation test for reconstruction accuracy for each
    % colour channel
    for chan = 1:3 % 3 colour channels
        tmp_sort = sort(abs(data.aver_all_Perm_chan(:,:,chan)), 'descend');
        tmp_act = abs(data.aver_all_chan(1,chan));
        tmp_result = find(tmp_sort>=tmp_act, 1, 'last'); % how many permuted accuracy is greater or equal to the actual accuracy
        if isempty(tmp_result)
            data.rnk_chan(:,chan) = 0;
        else
            data.rnk_chan(:,chan) = tmp_result;
        end
        data.p_val_Perm_chan(:,chan) = data.rnk_chan(:,chan)/data.accPermN; % the proportion of the values that are equal or greater than actual value
        data.p_val_Perm_chan_onetail(:,chan) = data.p_val_Perm_chan(:,chan)/2; % 1-tailed;
        alphaLevel = 0.05;
        data.CI_perm_chan(1,chan) = prctile(tmp_sort,100*alphaLevel/2); %CI lower bound
        data.CI_perm_chan(2,chan) = prctile(tmp_sort,100-100*alphaLevel/2); % CI upper bound
%         % CI based on a t distribution
%         critVals = tinv([data.alphaLevel/2, 1-data.alphaLevel/2], data.accPermN-1);
%         CI_t = mean(tmp_sort) + critVals * (std(tmp_sort)/sqrt(data.accPermN));
%         data.CI_t_perm_chan(1,chan) = CI_t(1);
%         data.CI_t_perm_chan(2,chan) = CI_t(2);
    end
end

%% plotting heatmaps of the test
% shows percetage of accurate descrimination (same pairs vs different
% pairs) in at each pixel.
[data.out_L] = heat_map_recon(data.recon_mat_sq(:,:,:,1:data.imNum),data.labout(1:data.imNum),1);
[data.out_A] = heat_map_recon(data.recon_mat_sq(:,:,:,1:data.imNum),data.labout(1:data.imNum),2);
[data.out_B] = heat_map_recon(data.recon_mat_sq(:,:,:,1:data.imNum),data.labout(1:data.imNum),3);

% each pixel's averaged accuracy across images
data.out_L_aver = squeeze(mean(data.out_L,1));
data.out_A_aver = squeeze(mean(data.out_A,1));
data.out_B_aver = squeeze(mean(data.out_B,1));

% flip the accuracy matrix left/right then average them to make the
% heatmap symmetric
flp = 1; 
if flp 
    data.out_flp_L = fliplr(data.out_L_aver);
    data.out_flp_A = fliplr(data.out_A_aver);
    data.out_flp_B = fliplr(data.out_B_aver);
    
    data.out_L_aver_sym = (data.out_L_aver + data.out_flp_L)/2;
    data.out_A_aver_sym = (data.out_A_aver + data.out_flp_A)/2;
    data.out_B_aver_sym = (data.out_B_aver + data.out_flp_B)/2;
end
 
vis = 0;
if vis
    lims=[0.1 0.9];
    %lims=[0 1];
    jet1=jet; jet1(1,:)=[0 0 0];
    %plot_heatmap(out_neut_L,lims,jet1,[1 2]) % the last input is optional, input to show heatmap of specific image
    plot_heatmap(data.out_L,lims,jet1);
    plot_heatmap(data.out_A,lims,jet1);
    plot_heatmap(data.out_B,lims,jet1);
end

% % plot heatmap after fliping
% figure
% imagesc(out_neutL_aver_sym,lims)
% set(gca,'Ticklength', [0 0],'YTickLabel',[],'XTickLabel',[], 'YAxisLocation', 'right')
% colormap(jet1)
% title ('Averged images')
% axis equal tight
% colorbar

end
function data = reconShp(group,subID,parameter,saveFileDir,workDir,conf,shapeFile,averFace,objAccPerm)
% Perception-based shape reconstruction of facial identities used in the 
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
%       shapeFile - shape information of the original images, the meaning
%                   of each dimension is: number of images x number of
%                   fidusial points x coordinates (x and y coordinates)
%       averFace - average face image used as the background for shape
%                  heatmap
%       objAccPerm - whether running permutation test for objective
%                    accuracy test
%
% Output:
%       data - a structure array containing all outputs

% Writtern by C.-H. Eric Chang, Feb 2020

%% Inputs
data.group = group;
data.subID = subID;
data.conf = conf;
data.shapeFile = shapeFile;
data.averFace = averFace;
data.objAccPerm = objAccPerm;

data.maxDim = parameter.maxDim; %20; % number of MDS to retain
data.permutN = parameter.permutN; %10000; % number of permuations to perfrom
data.minNumPix = parameter.minNumPix; %2; % minimum number of pixels to include as a signficiant dimension
data.q = parameter.q; %0.1; % criterion (Q) for the FDR correction
data.accPermN = parameter.accPermN; % number of permutations for the significance of objective accuracy

data.imNum = size(conf,1); % nubmer of images
cd(workDir);

%% Computing MDS
% computes visualization all identity included matrix loadings57 (ids X MDS dims) 
% and leave-one-out matrix for permuations loadings56 (ids X ids X MDS dims) 
% Important: loadings56 contains a procurstian alignment weights for each identity N
% at n X MDS_dims X n
% the last input can be replaced by [1:57] if only names have to be presented
[data.loadAll,data.loadLeaveOut,~,data.vari_expl,data.vari_exp_cum] = patMDS(data.conf,data.maxDim,0.0001);
%[data.loadAll,data.loadLeaveOut,~,data.vari_expl,data.vari_exp_cum]=patMDS(data.conf,data.maxDim,0.0001,data.imgFile); 
% [~,~,~,~,~]=patMDS(data.conf,data.maxDim,0.0001,[1:25]);

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

run = 1;
if run
    [data.p_All,data.CI]=ImClass_shape(data.shapeFile,data.loadAll,data.permutN,0); %p_happ60,CI_happy,
    outFile=[saveFileDir 'CI_All_' group num2str(subID) '_perm' num2str(data.permutN) '_' date];
    p_All = data.p_All;
    CI = data.CI;
    save(outFile,'p_All','CI'); %'p_happ57','CI_happy'
else
%     load([saveFileDir '/CI_All_s01_perm1000_24-Oct-2017.mat'])
end

%% FDR corrected iterations with leave-one-out
run = 1;
if run
    [data.p_LOut,~]=ImClass_shape(data.shapeFile,data.loadLeaveOut,data.permutN,1); %p_happ59
    outFile=[saveFileDir 'CI_LOut_' group num2str(subID) '_perm' num2str(data.permutN) '_' date];
    p_LOut = data.p_LOut;
    save(outFile,'p_LOut'); %'p_happ59'
else
%     load([saveFileDir '/CI_LOut_s01_perm1000_24-Oct-2017.mat'])
end
[data.outMatGen] = FDR_CI_sel_shape(data.p_LOut,data.q,data.minNumPix);

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
[data.recon_mat] = face_reconst_shape(data.outMatGen,data.loadLeaveOut,data.shapeFile);
% recon_mat: number of markers*2 (x+y coord) x number of images 

%% arranging back into row x column X color X dim array
data.recon_mat_sq = reshape(data.recon_mat,size(data.recon_mat,1)/2,2,size(data.recon_mat,2));
% recon_mat_sq: number of markers x 2 coord (x, y) x number of images

%% visualization of the shape
vis = 0;
imageN = 15; % visualize which image
if vis
    fig=figure;
    set(fig, 'Position', [100, 100, 800, 500]);
    subplot(1,2,1)
    scatter(data.recon_mat_sq(:,1,imageN), data.recon_mat_sq(:,2,imageN))
    title(['Reconstructed identity number ' num2str(imageN)])
    gname(1:82)
    subplot(1,2,2)
    scatter(shapeFile(imageN,:,1)', data.shapeFile(imageN,:,2)')
    title(['Original identity number ' num2str(imageN)])
    gname(1:82)
end

%% Save the reconstructed 57 unfamiliar faces (shape recon)
saveImg=0;
for i = 1:data.imNum
    data.ims_reconTp_shape{i,1} = data.recon_mat_sq(:,:,i);
end
if saveImg
    reconOut=fullfile(saveFileDir,['ims_reconTp_shape_' group num2str(subID) '_' date '.mat']);
    ims_reconTp_shape = data.ims_reconTp_shape;
    save(reconOut,'ims_reconTp_shape')
end

%% running objective test 
% reconstructed image compared with pairs (original image and all different
% images) based on Euclidean distance. If the original image is closer a
% score of 1 is awarded, if further a score of 0 is awarded. Then all the
% scores are averaged per image, and the grand average of all reconstructed
% faces is compared against 0.5 using t-test.
[~,data.p_val,data.aver_im,data.aver_all] = obj_test_shape(data.recon_mat_sq,data.shapeFile,0.05); 
if objAccPerm==1    
    loadLeaveOut = data.loadLeaveOut;
    maxDim = data.maxDim;
    imNum = data.imNum;
    outMatGen = data.outMatGen;
    disp('permutation test for reconstruction accuracy started')
    parfor k = 1:data.accPermN
        loadMatPerm = loadLeaveOut;
        for i = 1:maxDim
            for j = 1:imNum
                temp = loadMatPerm(:,i,j);
                temp(j) = [];
                temp(randperm(imNum-1)) = temp;
                loadMatPerm(setdiff(1:imNum,j),i,j) = temp;
            end
        end
        recon_matPerm = face_reconst_shape(outMatGen,loadMatPerm,shapeFile);
        recon_matPerm_sq = reshape(recon_matPerm,size(recon_matPerm,1)/2,2,size(recon_matPerm,2));
        
        [~,~,aver_im_Perm(:,k),aver_all_Perm(1,k)] = obj_test_shape(recon_matPerm_sq,shapeFile,0.05); 
    end
    disp('permutation test for reconstruction accuracy done')
    data.aver_im_Perm = aver_im_Perm;
    data.aver_all_Perm = aver_all_Perm;
    
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
end

%% Shape heatmap
vis = 0;
[data.heatmapShape_out, data.acc_shapePoints, data.acc_shapePts_flpaver]=heat_map_recon_shape(data.recon_mat_sq,data.shapeFile,data.averFace,vis);

end
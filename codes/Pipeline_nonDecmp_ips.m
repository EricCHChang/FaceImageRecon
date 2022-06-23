function Pipeline_nonDecmp_ips
% Run the non-decomposed reconstruction for each participant in the
% individual difference in older adults study (tested in person, using the
% aging study protocol)

% Create a for loop to run multiple participants 

%% Cleaning working space
clear all
close all
clc
% sca

%%
try
%% Inputs and Directories
% subject ID
group = 'ips'; 

% whether run super subject analysis (average dissimilarity matrices across
% subjects)
superSub = 0;
if superSub
    subs_all = {'SuperN9'};
    allSubs = {}; % subjects to be included in the super subject analysis
else
    subs_all = {'01'}; % change it depends on analyzing which subjects' data 
end

objAccPerm = 1; % do permutation test for objective accuracy -  yes:1; no:0
%saveAll = 1; % save all output - yes:1; no:0

% parameters
parameter.maxDim = 20; % number of MDS to retain
parameter.permutN = 1000; % number of permuations to perfrom
parameter.minNumPix = 10; % minimum number of pixels to include as a signficiant dimension
parameter.q = 0.1; % criterion (Q) for the FDR correction
parameter.accPermN = 1000; % number of permutations for the significance of objective accuracy

% directories
rootDir = cd;
rootDir = rootDir(1:end-length('analysis'));
workDir = [rootDir 'analysis'];
% p = mfilename('fullpath');
% p = p(1:end-27); % p = '/Users/ChiHsun/Dropbox/University of Toronto/PhD/Recon_mem_exp_patient/'; 
% workDir = [p 'analysis']; % the analysis folder
% saveFileDir = [p 'analysis/recon_surface_shape/s' subID]; % where to save the reconstruction output
% if ~exist(saveFileDir,'dir') 
%     mkdir(saveFileDir);
% end
confDataDir = [workDir '/dissimilarity_ips']; % where to read the confusability matrix
if ~exist(confDataDir,'dir')
    mkdir(confDataDir)
end

% load images used as inputs for non-decomposed reconstruction
load([rootDir 'ims_new.mat']) % load the 57 novel faces
imgFile = ims_new;
cd(workDir);

% % load background
% load('bck.mat') % load the indices of background (black) in the image

cd(workDir);

%% Start parallel pool
% parpool('default_jobmanager',12)

% Determine which platform the current Matlab is running on
if isunix && ~ismac
    osPlat = 'linux';
elseif ismac
    osPlat = 'mac';
end

if strcmp(osPlat,'linux')
    nMaxWorkers = 12; % number of maximum workers
elseif strcmp(osPlat,'mac')
    nMaxWorkers = 4;
end

poolobj = gcp('nocreate');
if isempty(poolobj)
    if strcmp(osPlat,'linux')
%         parpool('default_jobmanager',nMaxWorkers)
        parpool('local',nMaxWorkers)
    elseif strcmp(osPlat,'mac')
        parpool('local',nMaxWorkers)
    end 
end

%% Add required recon functions when running on the cluster
if strcmp(osPlat,'linux')
    addpath '/psyhome10/changc95/Recon_scripts'
end

%% Find the largest commond background among face images
saveOut = 0;
[outMask_nonDecmp,bck_common_nonDecmp] = commonbck(imgFile);

% Save the background index and outMask
if saveOut==1
    cd(workDir)
    save('bck_common_nonDecmp.mat','bck_common_nonDecmp')
    save('outMask_nonDecmp.mat','outMask_nonDecmp')
end

%% Applying the background mask on all images 
run = 0;
if run==1
    ims_unfam_nonDecmp_masked = applyingMask(imgFile,outMask_nonDecmp);
    save('ims_unfam_nonDecmp_masked.mat','ims_unfam_nonDecmp_masked');
else
    load('ims_unfam_nonDecmp_masked.mat')
end
imgFileMasked = ims_unfam_nonDecmp_masked;

%% Non-decomposed Reconstruction
for s = 1:length(subs_all)
    subID = subs_all{s}; % subject ID
    saveFileDir = [rootDir 'analysis/recon_nonDecmp_ips/' group subID '/']; % where to save the reconstruction output
    if ~exist(saveFileDir,'dir')
        mkdir(saveFileDir);
    end
    
    % check whether perception-based non-decomposed reconstruction has been conducted
    cd(saveFileDir)
    fileList = dir(['final_Tp_' group subID '*.mat']);
    savefileN = size(fileList,1);
    if savefileN==1 % if the results already existed and there is only one result mat file
        saveFileName = fileList.name;
        recon_percep = 1; % indicator that perception-based reconstruction has been conducted
    elseif savefileN>1 % if the results already existed and there are more than one result mat file
        % select the most recent completed result mat file
        for k = 1:savefileN
            saveFileDateMat(k,1) = fileList(k).datenum;
        end
        [~, saveFile_ind] = max(saveFileDateMat);
        saveFileName = fileList(saveFile_ind).name;
        recon_percep = 1; % indicator that perception-based reconstruction has been conducted
    elseif savefileN==0 % if the results don't exist
        recon_percep = 0; % indicator that perception-based reconstruction has NOT been conducted
    end
    cd(workDir)
    
    if recon_percep==0 % reconstruction for this subject has not been conducted
        % Compute (or load) dissimilarity matrix
        clear dismat conf
        dismat_filename = [confDataDir '/' group subID '_dismat_tp.mat'];
        if ~exist(dismat_filename,'file') % if the dissimilarity matrix not exist
            if superSub
                for k = 1:length(allSubs) % get each subject's dissimilarity matrix
                    dismatEach(:,:,k) = dismat_ips(group,allSubs{k}); % create from sub's responses
                end
                dismat = mean(dismatEach,3); % average dissimilarity matrix across subjects
                % save the average dissimilarity matrix
                out_dir = [rootDir 'analysis/dissimilarity_ips/'];
                outName = [group subID '_dismat_tp'];
                save([out_dir outName '.mat'],'dismat')
                dlmwrite([out_dir outName '.csv'],dismat)
            else
                dismat = dismat_ips(group,subID); % create from sub's responses
            end
        else
            disp(['Dissimilarity matrix of subject ' group subID ' already existed. Load it.'])
            load(dismat_filename) % load the dissimilarity matrix
        end
        conf = dismat/6; % divided by 6 so that each element is ranged between 0 and 1
        
        % Non-decomposed reconstruction
        disp(['subject ' group subID ' non-decomposed reconstruction started'])
        clear data_reconNonDecmp_Tp
        data_reconNonDecmp_Tp = reconNonDecmp(group,subID,parameter,saveFileDir,...
            workDir,conf,imgFileMasked,bck_common_nonDecmp,objAccPerm);
        % note: the images input here are already masked by the largest common
        % background across all images
        
        % Save the reconstruction output
        outFile = [saveFileDir 'final_Tp_' group num2str(subID) '_' date];
        if superSub
            save(outFile, 'data_reconNonDecmp_Tp','superSub','allSubs');
        else
            save(outFile, 'data_reconNonDecmp_Tp');
        end
        
        disp(['subject ' group subID ' non-decomposed reconstruction done'])
        
    else % reconstruction for this subject has been done
        disp(['subject ' group subID ' non-decomposed reconstruction started'])
        disp(['subject ' group subID ' non-decomposed reconstruction has been conducted'])
    end
    
end % end of all subjects

%% Shut down parallel pool and close MATLAB
poolobj = gcp('nocreate');
delete(poolobj);
%exit

%% Quit MATLAB
if strcmp(osPlat,'linux')
    exit
end

catch ME
    disp('Something wrong happened')    
    poolobj = gcp('nocreate');
    delete(poolobj);
    rethrow(ME)
    if strcmp(osPlat,'linux')
        exit
    end
    %exit
end

end


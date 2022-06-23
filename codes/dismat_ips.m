function dismat = dismat_ips(group,subID)
% Create the dissimilarity (confusability) matrix from Tp task
% Inputs: 
%       group - group ID ('ips')
%       subID - subject ID (e.g., '01')
%
% Output: 
%       dismat - dissimilarity ratings of all possible pairs among 57 faces;
%                the value ranges from 0 to 6, by which 0 means the most 
%                similar (least different) whereas 6 means the most 
%                dissimilar (most different)

% Written by C.-H. Eric Chang, Dec 2021

%%
dirc = mfilename('fullpath');
scrptName = mfilename;
root_dir = dirc(1:end-length(scrptName)-length('analysis')-1);
%root_dir = '/Users/ChiHsun/Dropbox/University of Toronto/PhD/Recon_mem_exp_patient/';

% read perceptual similarity rating data
[mat_act, ~, ~] = readData_ips(group,subID);
col_resp = 9; % which column is the subject's resposne
col_imgIndex1 = 11; % which column is the image file index on the left side
col_imgIndex2 = 12; % which column is the image file index on the right side
nCol = size(mat_act,2);

% change the scale from 1-7 to 0-6
% and convert the similarity scores to dissimilarity score
% which is: 0 means the most similar; 6 means the most dissimilar
mat_act(:,nCol+1) = 7 - mat_act(:,col_resp); % the 9th column is the rating

% % check if the readjustment is correct
% mat_act_sort = sortrows(mat_act,[7 8]);

% extract the dissimilarity rating for each pair
nStim = max(mat_act(:,col_imgIndex2));
dismat = zeros(nStim,nStim);
for i = 1:size(mat_act,1)
    ind_row = mat_act(i,col_imgIndex1);
    ind_col = mat_act(i,col_imgIndex2);
    dismat(ind_row,ind_col) = mat_act(i,nCol+1);
    dismat(ind_col,ind_row) = mat_act(i,nCol+1);
end

out_dir = [root_dir 'analysis/dissimilarity_ips/'];
outName = [group subID '_dismat_tp'];
    
if exist([out_dir outName '.mat'],'file')==0        
    % save the output
    save([out_dir outName '.mat'],'dismat')
    dlmwrite([out_dir outName '.csv'],dismat)
else
    disp(['Not saving dismat of ' group subID ' b/c dissimilarity matrix already existed.'])
end




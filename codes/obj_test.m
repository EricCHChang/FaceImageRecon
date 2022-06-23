function [out_acc_mat,p_val,aver_acc_im,aver_acc]=obj_test(recon_mat,ims,sig,kind)
%obj_test Computes objective test of the reconstruction using Euclidean
%distance and comparing each reconstruction with the original face and all
%other faces in pairs.
% Input:    recon_mat - matrix with reconstructed faces in LAB space:
%               double: rows x columns x channels x faces
%           ims - matrix of the original images: cell 120: rows x columns x
%               channles
%           sig - desired significance level
%           kind - 1: all 3 channels; 'L': L channel only; 'A': A channel
%                  only; 'B': B channel only.  Default is all
% Output:   out_ac_mat -  square matrix of confusability between all the
%           images. Values: 1 or 0
%           p_val - p-value of the ttest against 0.5 with H0 of unsuccessful
%           reconstruction.
%           aver_acc_im - average accuracy for each image
%           aver_acc - average accuracy of the entire batch.
if nargin<3
    sig=0.05;
    kind=1;
elseif nargin < 4
    kind=1;
end

if kind==1
    chans=1:3;
elseif strcmp(kind,'L')
    chans=1;
elseif strcmp(kind,'A')
    chans=2;
elseif strcmp(kind,'B')
    chans=3;
end

for i=1:size(recon_mat,4) % go through every reconstructed image
    Eucl_dist_true=(recon_mat(:,:,chans,i)-ims{i}(:,:,chans)).^2;
    Eucl_dist_true=sqrt(sum(Eucl_dist_true(:)));
    other_im=1:size(recon_mat,4);
    other_im(other_im==i)=[];
    for j=1:length(other_im)
        Eucl_dist_other=(recon_mat(:,:,chans,i)-ims{other_im(j)}(:,:,chans)).^2;
        Eucl_dist_other=sqrt(sum(Eucl_dist_other(:)));
        if Eucl_dist_other>Eucl_dist_true
            out_acc_mat(i,j)=1;
        else
            out_acc_mat(i,j)=0;
        end
    end
end
aver_acc_im=mean(out_acc_mat,2);
aver_acc=mean(aver_acc_im);
[~,p_val,~,~] = ttest(aver_acc_im,0.5,'Alpha',sig);

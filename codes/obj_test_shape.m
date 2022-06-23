function [out_acc_mat,p_val,aver_acc_im,aver_acc]=obj_test_shape(recon_mat,shapeFile,sig)
%%
%Arguments:
%       recon_mat - matrix of reconstruction. In recon_shape script, it
%                   reads the "recon_mat_sq", in which the dimensions are: 
%                   number of markers x 2 coord (x, y) x number of images
%       shapeFile - shape information of the original images. The
%                   dimensions are:
%                   number of images x number of fidusial points x coordinates (x and y coordinates)
%       sig - significant level

%%
recon_mat = permute(recon_mat,[3 1 2]);
% After the above step, the dimensions of recon_mat here are: 
% number of images x number of fiducial points x coordinates (x & y)

for i=1:size(recon_mat,1)
    Eucl_dist_true=squeeze((recon_mat(i,:,:)-shapeFile(i,:,:)).^2);
    Eucl_dist_true=sqrt(sum(Eucl_dist_true(:))); 
    other_im=1:size(recon_mat,1);
    other_im(other_im==i)=[];
    for j=1:length(other_im)
        Eucl_dist_other=squeeze((recon_mat(i,:,:)-shapeFile(other_im(j),:,:)).^2);
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

end
function [output,acc,acc_flpaver]=heat_map_recon_shape(recon_mat,shapeFile,averFace,vis)
%% Compute heatmap for shape reconstruction 
%Inputs:
%       recon_mat - reconstructed shape matrix. The dimensions are:
%                   number of fiducial points x 2(x&y) x number of images
%                   It reads the "recon_mat_sq"
%       shapeFile - shape information of the original images, the meaning
%                   of each dimension is: number of images x number of
%                   fidusial points x coordinates (x and y coordinates)
%       averFace - average all images to get an averaged face, to
%                  present it at the back and fiducial points are overlayed
%                  on it
% Outputs:
%       acc - reconstruction accuracy for each fiducial point, averaged
%             across all reconstructed images
%       acc_flpaver - average the accuracy of left/right fiducial points so
%                     that the heatmap looks symmetric

%%
recon_mat = permute(recon_mat,[3 1 2]);
% After the above step, the dimensions of recon_mat here are: 
% number of images x number of fiducial points x coordinates (x & y)

imNum=size(shapeFile,1); % number of images

% distance to the origin (0,0) of each fiducial point
for i=1:size(recon_mat,1) %number of images
    for j=1:size(recon_mat,2) %number of points
        output.recons_dist(i,j)=sqrt(recon_mat(i,j,1)^2+recon_mat(i,j,2)^2);
        output.shapes_dist(i,j)=sqrt(shapeFile(i,j,1)^2+shapeFile(i,j,2)^2);
    end
end

% standard deviation of each fiducial point's location (distance to origin) across images
for i=1:size(recon_mat,2)
    output.recons_std(i)=squeeze(std(output.recons_dist(:,i)));
    output.shapes_std(i)=squeeze(std(output.shapes_dist(:,i)));
end

% average location of each fiducial point across images
output.recons_pos=squeeze(mean(recon_mat,1)); 
output.shape_pos=squeeze(mean(shapeFile,1));  

%% Flipping left/right to make the size (standard deviation) of each points symmetric
indL = [2 4 6 8 10 12 16 19 22 25 27 30 32 35 38:48 60:65 74 76 78 81]';
indR = [3 5 7 9 11 13 17 20 23 26 28 31 33 36 49:59 66:71 75 77 79 82]';
indM = setdiff(1:size(recon_mat,2), [indL;indR]);

% recon_mat_flpaver = NaN(size(recon_mat,1),size(recon_mat,2),2);
% for i = 1:size(recon_mat,1) %number of images
%     for j = 1:length(indL) %number of symmetric points
%         jL = indL(j);
%         jR = indR(j);
%         x_averflp = (recon_mat(i,jL,1) + recon_mat(i,jR,1))/2;
%         y_averflp = (recon_mat(i,jL,2) + recon_mat(i,jR,2))/2;
%         recon_mat_flpaver(i,jL,1) = x_averflp;
%         recon_mat_flpaver(i,jR,1) = x_averflp;
%         recon_mat_flpaver(i,jL,2) = y_averflp;
%         recon_mat_flpaver(i,jR,2) = y_averflp;
%     end
% end

% i=1;
% jL=2;
% jR=3;
% 
% tmpLx = recon_mat(i,jL,1);
% tmpLy=recon_mat(i,jL,2);
% 
% tmpRx = recon_mat(i,jR,1);
% tmpRy=recon_mat(i,jR,2);
% 
% tmpx=(tmpLx+tmpRx)/2;
% tmpy=(tmpLy+tmpRy)/2;
% 
% tmp_dist(i,1)=sqrt(tmpx^2+tmpy^2)

output.recons_dist_flpaver = NaN(size(output.recons_dist,1),size(output.recons_dist,2));
for i = 1:length(indL)
    dist_averflp = (output.recons_dist(:,indL(i)) + output.recons_dist(:,indR(i)))/2;
    output.recons_dist_flpaver(:,indL(i)) = dist_averflp;
    output.recons_dist_flpaver(:,indR(i)) = dist_averflp;
end
output.recons_dist_flpaver(:,indM) = output.recons_dist(:,indM);

for i=1:size(recon_mat,2)
    output.recons_std_flpaver(i)=squeeze(std(output.recons_dist_flpaver(:,i)));
end


%% accuracy of each fiducial point
recon_mat_reshp=permute(recon_mat,[2,1,3]);
shapeFile_reshp=permute(shapeFile,[2,1,3]);
% number of fiducial points x number of images x 2(x&y)

for i=1:size(recon_mat_reshp,1) % number of fiducial points
    for j=1:size(recon_mat_reshp,2) % number of images
        recon_true=squeeze(recon_mat_reshp(i,j,:));
        shape_true=squeeze(shapeFile_reshp(i,j,:));
        dist_true=repmat(sum((recon_true-shape_true).^2),size(recon_mat_reshp,2)-1,1);
        shapes_others=squeeze(shapeFile_reshp(i,setdiff(1:size(recon_mat_reshp,2),j),:));
        recon_trues=repmat(recon_true',size(shapes_others,1),1);
        dist_others=sum((recon_trues-shapes_others).^2,2);
        % accuracy of each point for each image
        output.acc_pt(i,j)=sum(dist_true<=dist_others)/size(dist_true,1);
    end
end

% average accuracy of each point across memory targets
acc=mean(output.acc_pt,2); 

%% Fliping left/right to make the accuracy of each point symmetric
acc_flpaver = NaN(size(acc,1),1);
for i = 1:length(indL)
    acc_averflp = (acc(indL(i),1) +acc(indR(i),1))/2;
    acc_flpaver(indL(i),1) = acc_averflp;
    acc_flpaver(indR(i),1) = acc_averflp;
end
acc_flpaver(indM,1) = acc(indM,1);
    
%% Plot heatmap
if vis==1
    acc_scale=round(acc*64); % multiple by 64 to adjust the colour scale
    a=colormap(jet);
    %marg=[0.01 0.01];
    %figure
    % subplot_tight(1,2,1,marg)
    imagesc(averFace)
    %set(gca,'Ticklength', [0 0],'YTickLabel',[],'XTickLabel',[], 'YAxisLocation', 'right')
    axis equal tight
    colormap(gray)
    hold on
    SD=output.recons_std/max(output.recons_std);
    for i=1:size(output.shape_pos,1)
        scatter(output.shape_pos(i,1),output.shape_pos(i,2),100*SD(i),'filled',...
            'MarkerEdgeColor',a(acc_scale(i),:),'MarkerFaceColor',a(acc_scale(i),:))
        %     colorbar
    end
    % colorbar
end

%% Plot heatmap (after fliping)
if vis==1
    acc_flpaver_scale=round(acc_flpaver*64); % multiple by 64 to adjust the colour scale
    %a=colormap(jet);
    %marg=[0.01 0.01];
    figure
    % subplot_tight(1,2,1,marg)
    imagesc(averFace)
    %set(gca,'Ticklength', [0 0],'YTickLabel',[],'XTickLabel',[], 'YAxisLocation', 'right')
    axis equal tight
    colormap(gray)
    hold on
    SD_flpaver=output.recons_std_flpaver/max(output.recons_std_flpaver);
    for i=1:size(output.shape_pos,1)
        scatter(output.shape_pos(i,1),output.shape_pos(i,2),100*SD_flpaver(i),'filled',...
            'MarkerEdgeColor',a(acc_flpaver_scale(i),:),'MarkerFaceColor',a(acc_flpaver_scale(i),:))
        %     colorbar
    end
    % colorbar
end

function plot_heatmap(mat,clims,colors,indiv)
if nargin >3
    
    for i=1:length(indiv)
        outmat=squeeze(mat(indiv(i),:,:));
        figure
        imagesc(outmat,clims)
        colormap(colors)
        colorbar
        title(['image' num2str(indiv(i))])
        axis equal tight
        
        
    end
else
    outmat=squeeze(mean(mat,1)); % each pixel's averaged accuracy across images
    %figure
    imagesc(outmat,clims)
    set(gca,'Ticklength', [0 0],'YTickLabel',[],'XTickLabel',[], 'YAxisLocation', 'right')
    colormap(colors)
    title ('Averged images')
    axis equal tight
    colorbar
end
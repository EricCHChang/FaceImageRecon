function [labout]=convLab(imCellIn,test)
%This function converts cell array with images from RGB to LAB 
%imCellIn has to be a cell with size that equals the number of stimuli
%test - whether the force and back conversion is needed. 1 - needed, zero -
%not needed/
% Functions is written by DN. Based on MDS_neur_patt_v2

if nargin < 1;
    error(message('Not enough input'));
elseif nargin<2;
    test=0;
end
l=size(imCellIn); 
% cform_srgb2lab = makecform('srgb2lab');
for i=1:l(1)
    %a=double(imCellIn{i}); % This is incorrect, so removing it
%     a=applycform(a, cform_srgb2lab);
%     colorTransform = makecform('srgb2lab');
%     labout{i} = applycform(a, colorTransform);
    %labout{i}=RGB2Lab(a);
    labout{i}=rgb2lab(imCellIn{i}); %labout{i}=rgb2lab(a);
end
labout=labout';
if test
    rgb=[];
    lab=[];
    lab2=[];
    %cform_lab2srgb=makecform('lab2srgb');
    for i=1:l(1)
        rgb=cat(4,rgb,imCellIn{i});
        lab=cat(4,lab,labout{i});
        %lab2=cat(4,lab2,Lab2RGB(labout{i}));
        lab2=cat(4,lab2,lab2rgb(labout{i}));
%         colorTransform = makecform('lab2srgb');
%         lab2 = cat(4,lab2,applycform(labout{i}, colorTransform));
    end
    rgbMean=uint8(mean(rgb,4));
    %labMean=Lab2RGB(mean(lab,4));
    labMean=uint8(lab2rgb(mean(lab,4)));
%     colorTransform = makecform('lab2srgb');
%     labMean = applycform(mean(lab,4), colorTransform);
    %lab2Mean=mean(lab2,4);
    lab2Mean=uint8(mean(lab2,4));
    labDiff=labMean-rgbMean;
    lab2Diff=lab2Mean-rgbMean;
    imtool(rgbMean)
    imtool(labMean)
    imtool(lab2Mean)
    figure 
    hist(double(labDiff))
    title('Subtraction of converted average from the average RGB')
    figure 
    hist(double(lab2Diff))
    title('Subtraction of averaged individually converted files from the average RGB')
else
    disp('No test was requested')
end
        
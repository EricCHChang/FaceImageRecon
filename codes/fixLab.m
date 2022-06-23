function outFix=fixLab (imIn)
%% This function takes LAB converted images and converts them to 0-1 scale

dimN=ndims(imIn);
if dimN==3
 
    for i=1:size(imIn,3);
        temp=imIn(:,:,i);
        maxAbs=max(abs(temp(:)));
        imIn(:,:,i)=0.5+(temp/maxAbs)/2;
    end
elseif dimN==4
    for i=1:size(imIn,3)
        for j=1:size(imIn,4)
            temp=imIn(:,:,i,j);
            maxAbs=max(abs(temp(:)));
            imIn(:,:,i,j)=0.5+(temp/maxAbs)/2;
        end
    end
else
    error('Too many dimenstions')
end
outFix=imIn;
            
    
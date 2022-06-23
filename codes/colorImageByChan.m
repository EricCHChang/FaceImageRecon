function imOut=colorImageByChan(imIn,chan)
imOut=repmat(imIn,1,1,3);
switch chan
    case 1
for i=1:size(imIn,1)
    for j=1:size(imIn,2)
        if imIn(i,j)==0.5
            imOut(i,j,:)=0.5;
        elseif imIn(i,j)<0.5
            imOut(i,j,:)=0;
        else
            imOut(i,j,:)=1;
        end
    end
end
    case 2
for i=1:size(imIn,1)
    for j=1:size(imIn,2)
        if imIn(i,j)==0.5
            imOut(i,j,:)=0.5;
        elseif imIn(i,j)<0.5
            imOut(i,j,1)=1;
            imOut(i,j,2)=0;
            imOut(i,j,3)=0;
        else
            imOut(i,j,1)=0;
            imOut(i,j,2)=1;
            imOut(i,j,3)=0;
        end
    end
end
    case 3
for i=1:size(imIn,1)
    for j=1:size(imIn,2)
        if imIn(i,j)==0.5
            imOut(i,j,:)=0.5;
        elseif imIn(i,j)<0.5
            imOut(i,j,1)=1;
            imOut(i,j,2)=1;
            imOut(i,j,3)=0;
        else
            imOut(i,j,1)=0;
            imOut(i,j,2)=0;
            imOut(i,j,3)=1;
        end
    end
end
end
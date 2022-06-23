function [coef,all]=compContCoeff(matA,matB,desired_cont)
% compContCoeff computes coefficient to be for a CIs to be multiplied by to
% achieve the desired image contrast. For reconstruction.
%   matA - average matrix
%   matB - CI derived matrix
%   desired_cont - target contrast

contA=std(matA(:));
contB=std(matB(:));
covAB=cov(matA(:)',matB(:)');
covAB=covAB(1,2);
b=2*covAB;
a=contB^2;
c=contA^2-desired_cont^2;
% all(1,1)=(-b^2-sqrt(b^2-4*a*c))/(2*a)
% all(2,1)=(-b^2+sqrt(b^2-4*a*c))/(2*a)

all=roots([a b c]);
coef=max(all);
if imag(coef)==0
else
    all=roots([a b 0]);
    coef=max(all);
end
    
all(3)=a;
all(4)=b;
all(5)=c;



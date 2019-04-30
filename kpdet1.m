function detectionImage = kpdet1(I)
% KPDET Dectects the various features in an image 
%
% detectionImage = KPDET(I) where I is a grayscale image and detectionImage
% is a matrix of the same size as I with non-zero values indicating
% features. 
%
%
% This code is from CSC 262 Lab: Feature Detection
    
    %Setting gaussians for convolution
    gauss1 = gkern(1);
    gauss15 = gkern(1.5^2);
    gaussDeriv = gkern(1, 1);
    
    %Calculating the gradients 
    IXderiv = conv2(gauss1, gaussDeriv, I, 'same');
    IYderiv = conv2(gaussDeriv, gauss1, I, 'same');
    
    %Calculating the values for the matrix A
    IX2 = conv2(gauss15, gauss15, IXderiv.*IXderiv, 'same');
    IXY = conv2(gauss15, gauss15, IXderiv.*IYderiv, 'same');
    IY2 = conv2(gauss15, gauss15, IYderiv.*IYderiv, 'same');
    
    %Determinant, trace, and their quotient of A
    detA = (IX2.*IY2 - IXY.^2);
    trA = (IX2 + IY2);
    detOverTrace = (detA./trA);
    
    %Creating and thresholding the detection image
    thresholdValue = max(detOverTrace(:))/80;
    IMaxima = maxima(detOverTrace);
    detectionImage = detOverTrace;
    detectionImage(~IMaxima) = 0;
    detectionImage(detectionImage < thresholdValue) = 0;
end
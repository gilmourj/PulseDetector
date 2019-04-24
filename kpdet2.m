% KPDET detects interesting features in an image by finding areas with
% significant highly localized change in brightness in two different
% directions
%
% detectionI = kpdet(I) where I is a matrix of doubles representing an
% image and detectionI is a matrix of the same size containing the
% direction of the gradient at points where there was significant highly
% localized change in brightness in I, and zero everywhere else
function [detectionI] = kpdet2(I)
    % Create Gaussian kernels
    g = gkern(1); 
    gD = gkern(1,1); 
    g5 = gkern(1.5^2);
    
    % Calculate partial derivatives and blur with Gaussian
    Ix = conv2(g, gD, I, 'same'); 
    Iy = conv2(gD, g, I, 'same'); 
    Ix2 = conv2(g5, g5, Ix.*Ix, 'same'); 
    Iy2 = conv2(g5, g5, Iy.*Iy, 'same'); 
    Ixy = conv2(g5, g5, Ix.*Iy, 'same');
    
    % Calculate determinant, trace, and mean
    detI = Ix2.*Iy2 - Ixy.*Ixy; 
    traceI = Ix2 + Iy2; 
    meanI = detI./traceI;
    
    % Threshold image to show only strong local responses
    threshold = 6*10^-3; 
    M = maxima(meanI);
    
    % Blur with larger Gaussian and calculate orientation of gradient at
    % each detected feature
    g4 = gkern(4.5^2); 
    Ix = conv2(g4, g4, Ix, 'same'); 
    Iy = conv2(g4, g4, Iy, 'same'); 
    binaryI = M & (meanI > threshold); 
    orienI = atan2(Ix, Iy); 
    detectionI = orienI.*binaryI; 
end
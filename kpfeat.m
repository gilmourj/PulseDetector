% KPFEAT "describes" given keypoints in an image--so that they may be
% compared for matching.
%
% descriptors = kpfeat(img, keypoints) where img is a NxM array of doubles
% between 0 and 1 (inclusive) representing pixel brighnesses and keypoints
% is a NxM array of doubles with non-zero values at locations of detected
% keypoints and zeros otherwise. descriptors is a Kx64 matrix of doubles
% where K is the number of keypoints and every row of the matrix contains
% the pixel values of an 8x8 patch around the keypoint in img as a vector.
function [descriptors] = kpfeat(img, keypoints)

   downsampling_factor = 5;
   patch_size = 8;
   
    % find indices of keypoint
    [rowI, colI] = find(keypoints);
    
    % allocate space for result
    descriptors(size(rowI, 1), patch_size ^ 2) = 0;
    
    % blur and downsample input image
    gauss_5 = gkern(5^2);
    blurred_img = conv2(gauss_5, gauss_5, img, 'same');
    blurred_img = blurred_img(1:downsampling_factor:end, ...
        1:downsampling_factor:end);
   
    % loop through all features and extract a patch
    for k=1:size(rowI,1)
        
        row_center = floor(rowI(k)/downsampling_factor);
        col_center = floor(colI(k)/downsampling_factor);
        
        row_upper_left = row_center - 3;
        col_upper_left = col_center - 3;
        
        row_lower_right = row_center + 4;
        col_lower_right = col_center + 4;
        
        [maxRows, maxCols] = size(blurred_img);
        
        % check that patch does not fall outside image bounds
        if(col_upper_left < 1 || row_upper_left < 1 || ...
                row_lower_right > maxRows || col_lower_right > maxCols)
            descriptors(k,:) = NaN;
            continue;
        end
        
        % extract the patch
        patch = blurred_img(row_upper_left:row_lower_right, ...
            col_upper_left:col_lower_right);

        % normalize bias and gain of patch
        bias_norm_patch = patch - mean(patch(:));
        norm_patch = bias_norm_patch ./ std(bias_norm_patch(:));
        
        % store into array
        descriptors(k,:) = norm_patch(:);
    end  
end

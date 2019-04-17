%% CSC 262 Lab: Feature Matching

%% Introduction
% In this lab, we will implement an algorithm which matches features
% detected and described in previous labs. We do so in order to align two
% images that capture the same scene differently. We hope to learn how to
% determine the "best" match for a given keypoint and calculate the
% corresponding translation from a keypoint and its match in the other
% image. Using these keypoint matches, we will stitch two images of a
% waterfall together and observe the quality of their alignment.

%% Matching Features
% The first task is the identify matching pairs of keypoints. We do so
% using a notion of "distance" between feature descriptors (that is,
% patches). To start, we took one keypoint as reference arbitrarily and
% found its match; this process we later generalized for every keypoint.
%
% By distance, we mean the squared difference between the feature patches
% (themselves vectors). Below, we show the sorted feature distances between
% our reference feature and every keypoint in the other image (visualizing
% only the ten closest potential matches).

% PRELIMINARIES:
% load image
img1 = im2double( ...
    imread('/home/weinman/courses/CSC262/images/kpmatch1.png'));
img2 = im2double(...
    imread('/home/weinman/courses/CSC262/images/kpmatch2.png'));

% A. MATCHING SET-UP
% extract the keypoint locations 
locations1 = kpdet1(img1);
[rowLoc1, colLoc1] = find(locations1);

locations2 = kpdet1(img2);
[rowLoc2, colLoc2] = find(locations2);

% extract the keypoint descriptors
descriptors1 = kpfeat(img1, locations1);
descriptors2 = kpfeat(img2, locations2);

% B. FEATURE MATCHING
% take an arbitrary keypoint (that isn't NaN)
findex = 35;

% Extract the feature (the matrix row) 
currentFeatureD = descriptors1(findex,:);

% calculate the element-wise difference between every feature's values and
% your reference feature's values
% (pixel to pixel difference)
distance = descriptors2 - currentFeatureD;

% calculate the Euclidean distances between your chosen feature and all the
% features of the other (non-reference) image.
% (squared difference between vectors--row is summed)
% We don't take square root here bc thats expensive so we need to compare
% this to a squared threshold
euclidDist = sum(distance.^2,2);
[sortedEuclidDist, originalIndices] = sort(euclidDist);

% display bar chart of closest neighbors 
figure;
bar(sortedEuclidDist(1:10));
title('Distances to Top 10 Closest Features');
pause(0.5)

%% 
% Our reference feature appears to have a match--if only in the sense that
% it has a closest point (by our operating definition of distance). Notice
% further that there are two keypoints which are very close to our
% reference feature. On the one hand, this suggests a good match because
% these two potential matches are much closer to the reference than any
% other keypoint, even in the top ten. On the other hand, a specific match
% may be poor because the closest neighbor is not much closer than the
% second-closest neighbor. To avoid spurious matches, we ignore keypoints
% whose ratio of distance to closest neighbor to distance to
% second-closest neighbor exceeds a certain threshold. In this lab, we use
% Brown et al.'s recommendation of a threshold of 0.5. However, we use
% 0.5^2 because we consider squared differences as our notion of distance.

%% Alignment
% Now, we have matches. Our task is to align the images, which requires
% knowledge about how a keypoint in one image ought to be translated to
% cover its match. Given two keypoints (r1, c1) and (r2, c2) (r stands for
% row and c stands for column), we found the translation across rows via r2
% - r1 and translation across columns via c2 - c1. Additionally, because we
% are connecting corresponding points between images concatenated
% horizontally, we need to include an offset equal to the total number of
% columns in the image to the column translation value. Armed with these
% translation values for every pair, we can draw a line from each keypoint
% in one image to its match in the second image. Below, we show the
% result of this line-drawing.

% Calculate the estimated row translation 
translationRow = rowLoc2(originalIndices(1)) - rowLoc1(findex);

% Calculate the estimated column translation 
translationCol = colLoc2(originalIndices(1)) - colLoc1(findex);

% Concatenate reference image and the other image horizontally
imgConcat = [img1 img2];

% Display the concatenated image
% figure;
% imshow(imgConcat);
% title('Concatenated Image');
% hold on;
% %  draw a line from the reference feature location to its (probable) match
% colStart = colLoc1(findex);
% colEnd = colStart + size(img1, 2) + translationCol;
% % shift by size(img1, 2) to find corresponding col location in img2 (concat
% % horizontally)
% rowStart = rowLoc1(findex);
% rowEnd = rowStart + translationRow;
% 
% line([colStart colEnd], [rowStart rowEnd]); % draw the line btw features
% pause(0.5)

% syntax note: line([%movement horizontally (column change)%] [%movement
% vertically (row change)%])

% D. FEATURE MATCHING REDUX 
% preallocate Nx2 matrix for translation estimates
% for all the features in reference image
refKPCount = size(rowLoc1,1);
refKPtranslations = zeros(refKPCount, 2);

for i = 1:refKPCount
    % Extract the feature (the matrix row) 
    currentFeatureD = descriptors1(i,:);
    
    % if the current reference feature is invalid, set translation estimate
    % to [NaN Nan]
    if isnan(currentFeatureD(1))
        refKPtranslations(i, :) = [NaN NaN];
        continue;
    end
    
    % calculate the element-wise difference between every feature's values and
    % your reference feature's values
    % (pixel to pixel difference)
    distance = descriptors2 - currentFeatureD;
    
    % calculate the Euclidean distances between your chosen feature and all the
    % features of the other (non-reference) image.
    % (squared difference between vectors--row is summed)
    % We don't take square root here bc thats expensive so we need to compare
    % this to a squared threshold
    euclidDist = sum(distance.^2,2);
    [sortedEuclidDist, originalIndices] = sort(euclidDist);
    
    % If the nearest-neighbor squared distance ratio exceeds the squared
    % threshold set translation estimate to [NaN Nan]
    squaredThresh = 0.5^2;
    if (sortedEuclidDist(1) / sortedEuclidDist(2) > squaredThresh)
        refKPtranslations(i, :) = [NaN NaN];
        continue;
    end
    
    % Calculate the estimated row translation 
    translationRow = rowLoc2(originalIndices(1)) - rowLoc1(i);

    % Calculate the estimated column translation 
    translationCol = colLoc2(originalIndices(1)) - colLoc1(i);

    refKPtranslations(i, 1) = translationRow;
    refKPtranslations(i, 2) = translationCol;
end

% testing the loop made above (for part D)

% Display the concatenated image
pause(0.5);
figure;
imshow(imgConcat);
title('Feature Matches on Concatenated Image');
hold on;
for j = 1:size(refKPtranslations, 1)
    if isnan(refKPtranslations(j,1))
        continue
    end
    colStart = colLoc1(j);
    colEnd = colStart + size(img1, 2) + refKPtranslations(j, 2);
    rowStart = rowLoc1(j);
    rowEnd = rowStart + refKPtranslations(j, 1);
    line([colStart colEnd], [rowStart rowEnd]);
end
pause(0.5);

%%
% For alignment, however, we need to summarize all of these translations
% into one vector so that we may transform every pixel in the same way.
% Since we are using the least squares approximation of translation
% alignment, the best overall translation vector estimate is the mean of
% all of the row translations and the mean of the column translations
% encapsulated in a single vector. In our case, this vector was [-1 142]
% (note that we rounded the mean values to work better with our stitching
% algorithm). The standard deviation associated with this estimation
% process was 1.2799 for the row translations and 3.8073 for the column
% translations. We predict that, given these values, our alignment will be
% fairly strong if not perfect. Some standard deviation exists among the
% row and column translations, which means that our estimate will not
% capture the translation of outliers--leading to blurriness in the
% outlined image. Our tentative hypothesis is that the alignment
% column-wise will be worse than the alignment vertically given the higher
% standard deviation.
%
% Next, we performed the alignment on the images using a provided procedure
% 'stitch'--which yields the below image.

% E. Alignment
% Sort row Translations in ascending order
sortedTransRows = sort(refKPtranslations(:, 1));
% Sort column Translations in ascending order
sortedTransCols = sort(refKPtranslations(:, 2));


% Filter out NaN's from row and column Translations
sortedTransRows = sortedTransRows(1:(find(isnan(sortedTransRows), 1))-1);
sortedTransCols = sortedTransCols(1:(find(isnan(sortedTransCols), 1))-1);

% Take the (rounded) mean of the row translations and column translations
% to make an overall translation estimate
OverallTrans(1) = round(mean(sortedTransRows));
OverallTrans(2) = round(mean(sortedTransCols));

% calculate standard deviation of translations
OvervallTransStd(1) = std(sortedTransRows);
OvervallTransStd(2) = std(sortedTransCols);

% Stitch images together
% Make overall translation into integers so that it works with stitch
%stitched = stitch(img1, img2, [OverallTrans(1) OverallTrans(2)]);
stitched = stitch(img2, img1, [OverallTrans(1) OverallTrans(2)]);

% Display stitched image
figure;
imshow(stitched);
title('Stitched Image');
pause(0.1)

%%
% Overall, we are satisfied with the quality of our alignment. The image is
% blurry, but the major features of the landscape are aligned--including
% the main rock formation on the face of the hill and the shoreline itself.
% The alignment appears to be better in the lower half of the stitched
% image (in particular: the bushes, rocks, and water). We believe that
% alignment succeeds here due to the large number of feature matches in
% that area. A larger number of match pairs has a stronger effect on our
% overall translation vector estimate (because it is a mean) and thus
% favors those translations in our alignment procedure. On the other hand,
% alignment is poor at the the top of the middle waterfall because there
% was only one match--and the contrast between the land and air makes
% vertical and horizontal alignment mistakes more obvious. We could not
% reach any firm conclusion about whether alignment vertically or
% horizontally was stronger.

%% Conclusion
% This lab capped our endeavor in relating contents of related images.
% Building on our previous work in detection and description, we were able
% to successfully identify feature matches and use those matches to align
% two images. We learned how to classify strong matches based on closest
% distance and nearest-neighbor distance ratio. Then, we calculated
% translation vectors in an effort to determine how a keypoint in one
% image corresponded to its match in the other image. Finally, we stitched
% two images together based on an estimate of the "best" translation and
% observed the quality of our alignment. Although we were largely satisfied
% with the results of the alignment--that is, major features lined up in
% the resulting image--we found noticeable errors in both vertical and
% horizontal alignment.

%% Acknowledgements
% Code and text of the CSC 262 Lab: Feature Matching and associated
% materials, written by Jerod Weinman, informed our completion of the lab
% exercises. We also relied on keypoint detection and description scripts
% written by us in collaboration with former lab partners. We relied
% indirectly on formulas and suggestions from the Szeliski textbook
% (Richard Szeliski, Computer Vision: Algorithms and Applications,
% Springer, Electronic Draft, September 2010) and Brown et al. The images
% kpmatch* were taken by Jerod Weinman at Parque Nacional do Iguacu in
% Brazil, the images lectkpmatch* were taken by Jerod Weinman in
% Marseilles, France.
%% CSC262 Final Project
% This final project for CSC 262 seeks to reproduce results for detecting
% pulse from head motion from Balakrishnan et al. 2013 
%
% Input: video of head
% Output: pulse rate
 
% Steps of the algorithm:
% 1. Read in video and store frames
% 2. Get rid of extra space around face and block out eyes
% 3. Detect keypoints in first frame and save results of detections
% 4. Match keypoints between first frame and all other frames
% 5. Calculate change in location of each keypoint between frames
% 6. Discard unstable keypoints
% 7. Filter out lowest and highest frequencies in the frequency domain
% 8. Run PCA Decomposition on filtered signals 
% 9. Select most periodic signal
% 10. Calculate pulse
 
%%
% Prep
run ~/startup.m
 
%% Initialize Variables
vidReader1 = VideoReader('zoe_120fps.mp4');
numFrames = vidReader1.NumberOfFrames;
 
% Re-create object so we can use 'hasFrame' after using 'NumberOfFrames'
vidReader = VideoReader('zoe_120fps.mp4');
 
% Save dimensions used to trim frames (found manually, since we didn't use
% a face detection algorithm)
trimmedTop1 = 255;
trimmedBottom1 = 755;
trimmedLeft1 = 800;
trimmedRight1 = 1150;
 
% Save parameters to block out section around eyes
eyeTop1 = 170;
eyeBottom1 = 320;
 
%% Detect Features
% Read, crop, and block out eyes in first frame
frame1 = rgb2gray(readFrame(vidReader));
trimmedFrame1 = frame1(trimmedTop1:trimmedBottom1,trimmedLeft1:trimmedRight1);
trimmedFrame1(eyeTop1:eyeBottom1,:) = NaN;
 
%%
% Detect features in first frame
kps = kpdet1(trimmedFrame1);
[r1, c1] = find(kps > 0);
numKeypoints = size(r1,1);
 
% Display the trimmed first frame with the detected keypoints
figure;
imshow(trimmedFrame1,[]);
hold on;
for i = 1:size(find(kps > 0),1)
    plot(c1(i), r1(i), 'r+');
    hold on;
end
title('Trimmed First Frame with Detected Keypoints (Subject 1)');
 
% Create matrices to store keypoint locations in all frames
rowLocations = zeros(numKeypoints,numFrames);
colLocations = zeros(numKeypoints,numFrames);
rowLocations(:,1) = r1;
colLocations(:,1) = c1;
 
% Create matrix for feature descriptors from all frames
featDesc = zeros(numKeypoints,64,numFrames, 'uint8');
 
% Get feature descriptors for first frame
featDesc(:,:,1) = kpfeat(trimmedFrame1,kps);
 
% Initialize parameters for feature matching process
maxVertDisp = 10;
maxHorizDisp = 10;
 
% Allocate matrix for SSD values
SSDs = zeros(10,10);
 
% Read and process all the frames
k = 2;
while(hasFrame(vidReader))
    
    % Read, trim, and block out eyes in next frame
    frame = rgb2gray(readFrame(vidReader));
    trimmedFrame = frame(trimmedTop1:trimmedBottom1, ... 
        trimmedLeft1:trimmedRight1);
    trimmedFrame(eyeTop1:eyeBottom1,:) = NaN;
    
    % Find matches for all the features from the first frame
    for j = 1:size(featDesc,1)
        
        % Find upper right corner of box in which to look for feature
        ulCornerCol = max(colLocations(j,k-1) - 5,1);
        ulCornerRow = max(rowLocations(j,k-1) - 5,1);
        
        % Search for feature match in 10x10 pixel box
        for row = 1:maxVertDisp
            for col = 1:maxHorizDisp
                rowStart = ulCornerRow+row;
                rowEnd = ulCornerRow+row+7;
                colStart = ulCornerCol+col;
                colEnd = ulCornerCol+col+7;
                
                % Make sure that the patch doesn't extend past the edge of
                % the image
                if(rowEnd <= size(trimmedFrame,1) ... 
                        && colEnd <= size(trimmedFrame,2))
                    patch = trimmedFrame(rowStart:rowEnd, ...
                        colStart:colEnd);
                    patchVec = patch(:);
                    SSDs(row,col) = sum((featDesc(j,:,k-1) ...
                        - patchVec').^2);
                else
                    % If the patch goes off the image, set the value for
                    % the match to NaN
                    SSDs(row,col) = NaN;
                end
            end
        end
        
        % Take the feature with the lowest SSD value as the best match
        minSSD = min(SSDs(:));
        if(~isnan(minSSD))
            
            % Get indices of the best SSD value so we can find the location
            % of the patch that had that value
            [r,c] = find(SSDs <= minSSD);
            
            % Get location of the center of that patch within our box
            rowLocations(j,k) = ulCornerRow+r(1)+3;
            colLocations(j,k) = ulCornerCol+c(1)+3;
            
            % Extract the patch and save it in the feature descriptors
            % matrix
            p = trimmedFrame((ulCornerRow+r):(ulCornerRow+r+7), ...
                (ulCornerCol+c):(ulCornerCol+c+7));
            featDesc(j,:,k) = p(:);
        else
            % If all the SSDs are NaN, set best feature patch to NaN
            rowLocations(j,k) = NaN;
            colLocations(j,k) = NaN;
            featDesc(j,:,k) = NaN;
        end
    end
    
    % Increment counter
    k = k + 1;
end
 
%%
figure;
for i=1:numKeypoints
    plot(rowLocations(i,:));
    hold on;
end
title('Vertical Movement of Keypoints Across Frames (Subject 1)');
xlabel('Frame');
ylabel('Row Location of Feature');
 
%% Track Keypoint Movement
 
% Shift keypoint position matrix to the left by one column
pos2 = zeros(size(rowLocations,1),size(rowLocations,2) - 1);
pos2(:,:) = rowLocations(:,2:end);
 
% Subtract shifted matrix from original matrix to calculate distance moved
% between frames for each keypoint
amtMoved = abs(rowLocations(:,1:end - 1) - pos2);
 
% Find the maximum amount moved for each keypoint and remove NaN values
maxAmtMoved = max(amtMoved,[],2);
maxAmtMovedFiltered = maxAmtMoved(~isnan(maxAmtMoved));
 
% Find the mode of the maximums, and remove all keypoints whose max amount
% moved is greater than the mode
modeMaxAmtMoved = mode(maxAmtMovedFiltered);
indToKeep = find(maxAmtMovedFiltered < modeMaxAmtMoved);
colLocFilt = colLocations(indToKeep,:);
rowLocFilt = rowLocations(indToKeep,:);
 
%%  Filter Frequencies
frameRate = 120;
T = 1/frameRate;
t = (0:numFrames-1)*T;
 
% Find indices of the bounds of the frequencies that we want to keep
lowerBound = floor(numFrames*(.75/120));
upperBound = ceil(numFrames*(5/120));
 
% Compute the Fourier transform of all of the signals
fourierTrans = fft(rowLocFilt,[],2);
 
% Set frequencies outside of our bounds to zero
fourierTrans(:,1:lowerBound) = 0+0i;
fourierTrans(:,upperBound:end) = 0+0i;
 
% Reconstruct signal using inverse Fourier transform
filteredSignal = real(ifft(fourierTrans,[],2));
 
%% PCA Decomposition
 
% Compute covariance matrix
covMat = cov(filteredSignal');
 
% Compute eigenvectors for covariance matrix
[eigVecs,eigVals] = eigs(covMat);
numEigVecs = size(eigVecs,2);
 
% Calculate position signals by projecting time-series onto eigenvectors
posSignals = zeros(numEigVecs,numFrames);
posTimeSeries = filteredSignal';
for i = 1:numEigVecs
    for j = 1:numFrames
        posSignals(i,j) = dot(filteredSignal(:,j),eigVecs(:,i));
    end
end
 
%% Signal Selection
 
% Allocate space to store periodicity values
periodicities = zeros(size(posSignals,1),1);
 
% Calculate periodicity of each signal
for i = 1:size(posSignals,1)
    
    % Calculate the Fourier transform of the signal
    sigFourier = fft(posSignals(i,:));
    absFourierTrans = abs(sigFourier);
    
    % Find the index of the maximal frequency
    [m,maxInd] = max(absFourierTrans);
    
    % Calculate the maximal frequency and its first harmonic
    freq = frameRate * (maxInd/numFrames);
    harmonic = 2 * freq;
    
    % Find index of first harmonic
    harmInd = floor(numFrames*(harmonic/120));
    
    % Calculate the perioditicy of the signal using those values
    periodicities(i,1) = (absFourierTrans(maxInd) + ... 
        absFourierTrans(harmInd)) / sum(absFourierTrans,2);
end
 
% Find the maximum periodicity of all the signals
[maxPer, maxPerInd] = max(periodicities);
 
% Calculate the maximal frequency of the signal with the highest
% periodicity
[maxF,maxFreqInd] = max(abs(fft(posSignals(maxPerInd,:))));
maxFreq = frameRate * (maxFreqInd/numFrames);
 
% Display plot of signal with highest periodicity
figure;
plot(posSignals(maxPerInd,:));
title('Position Signal with the Highest Periodicity (Subject 1)');
xlabel('Frame');
ylabel('Position');
 
 
% Calculate the pulse rate
pulseRate = 60 / maxFreq;
 
%% Acknowledgements
% We completely this project with the help of Jerod Weinman, and the
% kpdet.m and kpfeat.m were written with partners on previous labs.
 
%% Citations
% Balakrishnan, G., Durand, F., & Guttag, J. (2013). Detecting pulse from
%   head motions in video. CVPR.
% "fft: Fast Fourier transform." MathWorks, 2019,
%   https://www.mathworks.com/help/matlab/ref/fft.html
% Krishnamurthy, R. Video Processing in Matlab. MathWorks, 2019,
%   https://www.mathworks.com/videos/video-processing-in-matlab-68745.html
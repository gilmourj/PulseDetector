%% CSC262 Final Project

% Input: video of head
% Output: pulse rate

% Steps:
% 1. Read in video and store frames
% 2. Get rid of extra space and block out eyes
% 3. Detect keypoints in each frame and save results of detections
% 4. Match keypoints between frames
% 5. Calculate change between frames
% 6. Get rid of unstable keypoints
% 7. Some complicated math to figure out how much head moves overall??
% 8. Use this info to calculate pulse rate??

%%
% Prep
run ~/startup.m

%% Initialize Variables
vidReader1 = VideoReader('zoe_120fps.mp4');
numFrames = vidReader1.NumberOfFrames;

% Re-create object so we can use 'hasFrame' after using 'NumberOfFrames'
vidReader = VideoReader('zoe_120fps.mp4');

% Save dimensions used to trim frames
trimmedHeight = 500;
trimmedWidth = 371;
eyeTop = 170;
eyeBottom = 320;

% Save parameters to block out section around eyes
trimmedTop = 255;
trimmedBottom = 755;
trimmedLeft = 800;
trimmedRight = 1150;

%% Detect Features
% Read and process first frame
frame1 = readFrame(vidReader);
frame1 = rgb2gray(frame1);
trimmedFrame1 = frame1(trimmedTop:trimmedBottom,trimmedLeft:trimmedRight);
trimmedFrame1(eyeTop:eyeBottom,:) = NaN;

% Detect features
kps = kpdet1(trimmedFrame1);
[r1, c1] = find(kps > 0);
numKeypoints = size(r1,1);

% create matrices to store keypoint locations in all frames
rowLocations = zeros(numKeypoints,numFrames);
colLocations = zeros(numKeypoints,numFrames);
rowLocations(:,1) = r1;
colLocations(:,1) = c1;

% create matrix for feature descriptors from all frames
featDesc = zeros(numKeypoints,64,numFrames, 'uint8');

% get feature descriptors for first frame
featDesc(:,:,1) = kpfeat(trimmedFrame1,kps);

% Initialize parameters for feature matching process
maxVertDisp = 10;
maxHorizDisp = 10;

% Allocate matrix for SSD values
SSDs = zeros(10,10);

% Read and process all the frames
k = 2;
while(hasFrame(vidReader))
    
    % Read and process frame
    frame = readFrame(vidReader);
    frame = rgb2gray(frame);
    trimmedFrame = frame(trimmedTop:trimmedBottom,trimmedLeft:trimmedRight);
    trimmedFrame(eyeTop:eyeBottom,:) = NaN;
    
    % Find matches for all the features from the first frame
    for j = 1:size(featDesc,1)
        ulCornerCol = max(colLocations(j,k-1) - 5,1);
        ulCornerRow = max(rowLocations(j,k-1) - 5,1);
        
        % Search for feature match in 10x10 pixel box
        for row = 1:maxVertDisp
            for col = 1:maxHorizDisp
                rowStart = ulCornerRow+row;
                rowEnd = ulCornerRow+row+7;
                colStart = ulCornerCol+col;
                colEnd = ulCornerCol+col+7;
                if(rowEnd <= size(trimmedFrame,1) ... 
                        && colEnd <= size(trimmedFrame,2))
                    patch = trimmedFrame(rowStart:rowEnd, ...
                        colStart:colEnd);
                    patchVec = patch(:);
                    SSDs(row,col) = sum((featDesc(j,:,k-1) ...
                        - patchVec').^2);
                else
                    SSDs(row,col) = NaN;
                end
            end
        end
        
        % Take the feature with the lowest SSD value as the best match
        minSSD = min(SSDs(:));
        if(~isnan(minSSD))
            [r,c] = find(SSDs <= minSSD);
            
            % Get location of the center of the best patch within our box
            rowLocations(j,k) = ulCornerRow+r(1)+3;
            colLocations(j,k) = ulCornerCol+c(1)+3;
            
            % Save best patch to feature descriptors matrix
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
% Visualization of movement
figure;
for i=1:numKeypoints
    plot(colLocations(i,:),rowLocations(i,:));
    hold on;
end

%%
figure;
for i=1:numKeypoints
    plot(rowLocations(i,:));
    hold on;
end
title('Vertical Movement of Keypoints Across Frames (Z)');
xlabel('Frame');
ylabel('Row Location of Feature');

%%
figure;
imshow(trimmedFrame,[]);
hold on;

for i = 1:size(find(kps > 0),1)
    plot(c1(i), r1(i), 'r+');
    hold on;
end

%% Calculate Pulse

% Discard the least stable keypoints
pos2 = zeros(size(colLocations,1),size(colLocations,2) - 1);
pos2(:,:) = colLocations(:,2:end);
amtMoved = abs(colLocations(:,1:end - 1) - pos2);
maxAmtMoved = max(amtMoved,[],2);

maxAmtMovedFiltered = maxAmtMoved(~isnan(maxAmtMoved));

modeMaxAmtMoved = mode(maxAmtMovedFiltered);

% only keep the keypoints whose max amount moved is less than the mode
indToKeep = find(maxAmtMovedFiltered < modeMaxAmtMoved);
colLocFilt = colLocations(indToKeep,:);
rowLocFilt = rowLocations(indToKeep,:);

%%
Fs = 120;
T = 1/Fs;
L = numFrames;
t = (0:L-1)*T;
figure;
plot(1000*t(1:numFrames),rowLocFilt(8,1:numFrames))
title('Signal')
xlabel('t (milliseconds)')
ylabel('X(t)')

%%

% Find indices of the bounds of the frequencies that we want to keep
lowerBound = floor(L*(.75/120));
upperBound = ceil(L*(5/120));

% Compute the Fourier transform of all of the signals
Y = fft(rowLocFilt,[],2);

% Set frequencies outside of our bounds to zero
Y(:,1:lowerBound) = 0+0i;
Y(:,upperBound:end) = 0+0i;

% P2 = abs(Y/L);
% P1 = P2(1:L/2+1);
% P1(2:end-1) = 2*P1(2:end-1);
% f = Fs*(0:(L/2))/L;
% figure;
% plot(f,P1) 
% title('Single-Sided Amplitude Spectrum of X(t)')
% xlabel('f (Hz)')
% ylabel('|P1(f)|')

% Reconstruct signal using inverse Fourier transform
filteredSignal = real(ifft(Y,[],2));
%%
% figure;
% plot(real(filteredSignal));

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

% Compute periodicity of each signal

% Allocate space to store periodicity values
periodicities = zeros(1,size(posSignals,1));

for i = 1:size(posSignals,1)
    absFourierTrans = abs(fft(posSignals(i,:)));
    maxFreq = max(absFourierTrans);
    
end

%% Citations
% Krishnamurthy, R. Video Processing in Matlab. MathWorks, 2019,
% https://www.mathworks.com/videos/video-processing-in-matlab-68745.html
% FFT help: https://www.mathworks.com/help/matlab/ref/fft.html



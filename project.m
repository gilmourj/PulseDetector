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
%% Read in Video
vidReader1 = VideoReader('zoe_120fps.mp4');
numFrames = vidReader1.NumberOfFrames;

vidReader = VideoReader('zoe_120fps.mp4');
%%
% preallocate space to store frames

trimmedHeight = 500;
trimmedWidth = 371;
eyeTop = 170;
eyeBottom = 320;

trimmedTop = 255;
trimmedBottom = 755;
trimmedLeft = 800;
trimmedRight = 1150;

%% Detect Features
% preallocate space for feature detections
frame1 = readFrame(vidReader);
frame1 = rgb2gray(frame1);
trimmedFrame1 = frame1(trimmedTop:trimmedBottom,trimmedLeft:trimmedRight);
trimmedFrame1(eyeTop:eyeBottom,:) = NaN;
kps = kpdet1(trimmedFrame1);

% find indices of keypoints
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
maxVertDisp = 10;
maxHorizDisp = 10;

% create matrix for SSD values
SSDs = zeros(10,10);

k = 2;
while(hasFrame(vidReader))
   frame = readFrame(vidReader);
   frame = rgb2gray(frame);
   trimmedFrame = frame(trimmedTop:trimmedBottom,trimmedLeft:trimmedRight);
   trimmedFrame(eyeTop:eyeBottom,:) = NaN;
   
     for j = 1:size(featDesc,1)
        ulCornerCol = max(colLocations(j,k-1) - 5,1);
        ulCornerRow = max(rowLocations(j,k-1) - 5,1);

        for row = 1:maxVertDisp
           for col = 1:maxHorizDisp
               rowStart = ulCornerRow+row;
               rowEnd = ulCornerRow+row+7;
               colStart = ulCornerCol+col;
               colEnd = ulCornerCol+col+7;
               if(rowEnd <= size(trimmedFrame,1) && colEnd <= size(trimmedFrame,2))
                   patch = trimmedFrame(rowStart:rowEnd, ...
                   colStart:colEnd);
                   patchVec = patch(:);
                   SSDs(row,col) = sum((featDesc(j,:,k-1) - patchVec').^2);
               else
                   SSDs(row,col) = NaN;
               end
           end
        end

        minSSD = min(SSDs(:));
        if(~isnan(minSSD))
            [r,c] = find(SSDs <= minSSD);

            % getting the location of the center of the best patch within our box
            rowLocations(j,k) = ulCornerRow+r(1)+3;
            colLocations(j,k) = ulCornerCol+c(1)+3;

            % save best patch to feature descriptors matrix
            p = trimmedFrame((ulCornerRow+r):(ulCornerRow+r+7), ...
                (ulCornerCol+c):(ulCornerCol+c+7));
            featDesc(j,:,k) = p(:);
        else
            rowLocations(j,k) = NaN;
            colLocations(j,k) = NaN;
            featDesc(j,:,k) = NaN;
        end
     end
    k=k+1;
end





%%
% Visualization of movement
figure;
for i=1:numKeypoints
    plot(colLocations(i,:),rowLocations(i,:));
    hold on;
end

%plot(colLocations(83,:),rowLocations(83,:));
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
%%
% 
% for i=1:numKeypoints
%     plot(rowLocations(i,:));
%     hold on;
% end

%%

% discard the least stable keypoints
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

% filter out frequencies that are too high or too low
% y = bandpass(colLocFilt(1,:),[0.75,5]);

% [b,a] = butter(5,[0.75,5]);
% filt = filter(b,a,colLocFilt(1,:));

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

lowerBound = floor(L*(.75/120));
upperBound = ceil(L*(5/120));

Y = fft(rowLocFilt(1,:));
Y(1:lowerBound) = 0+0i;
Y(upperBound:end) = 0+0i;

P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = Fs*(0:(L/2))/L;
figure;
plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

filteredSignal = ifft(Y);
%%
figure;
plot(real(filteredSignal));

%%
covMat = cov(rowLocFilt);
meanPos = mean(rowLocFilt, 2);

[eigVecs,eigVals] = eigs(covMat);

posSignals = zeros(6,size(rowLocFilt,1));
posTimeSeries = rowLocFilt';
for i = 1:6
    for j = 1:size(rowLocFilt,1)
        posSignals(i,j) = dot(rowLocFilt(j,:)',eigVecs(:,i));
    end
end

%% Citations
% Krishnamurthy, R. Video Processing in Matlab. MathWorks, 2019,
% https://www.mathworks.com/videos/video-processing-in-matlab-68745.html
% FFT help: https://www.mathworks.com/help/matlab/ref/fft.html



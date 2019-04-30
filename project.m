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
vidReader = VideoReader('zoe_120fps.mp4');
numFrames = 0;

%%
% preallocate space to store frames
frames = zeros(1080,1920,200);

for i = 1:200
   numFrames = numFrames + 1;
   %frame = imrotate(readFrame(vidReader),-90);
   frame = readFrame(vidReader);
   frame = rgb2gray(frame);
   frames(:,:,i) = frame;
end


frameHeight = size(frames,1);
frameWidth = size(frames,2);

%% Process Video

% preallocate space for new smaller frames
% trimmedHeight = 311;
% trimmedWidth = 227;
% eyeTop = 54;
% eyeBottom = 140;

trimmedHeight = 646;
trimmedWidth = 371;
eyeTop = 170;
eyeBottom = 320;

framesTrimmed = zeros(trimmedHeight,trimmedWidth,numFrames);

for frame = 1:numFrames
    % get rid of extra space around face and store in new array
    framesTrimmed(:,:,frame) = frames(255:900,780:1150,frame);
    
    % set area around eyes to NaN
    framesTrimmed(eyeTop:eyeBottom,:,frame) = NaN;
end

%% Detect Features
% preallocate space for feature detections
kps = zeros(trimmedHeight,trimmedWidth,numFrames);

for frame = 1:numFrames
    kps(:,:,frame) = kpdet1(framesTrimmed(:,:,frame));
end

% find indices of keypoints
[r1, c1] = find(kps(:,:,1) > 0);

numKeypoints = size(r1,1);

% create matrices to store keypoint locations in all frames
rowLocations = zeros(numKeypoints,numFrames);
colLocations = zeros(numKeypoints,numFrames);

rowLocations(:,1) = r1;
colLocations(:,1) = c1;
%%
figure;
imshow(framesTrimmed(:,:,1),[]);
hold on;


for i = 1:size(find(kps(:,:,1) > 0))
    plot(c1(i), r1(i), 'r+');
end

%% Match Features

% create matrix for feature descriptors from all frames
featDesc = zeros(numKeypoints,64,numFrames);

% get feature descriptors for first frame
featDesc(:,:,1) = kpfeat(framesTrimmed(:,:,1),kps(:,:,1));
maxVertDisp = 10;
maxHorizDisp = 10;

% create matrix for SSD values
SSDs = zeros(10,10);

for k = 2:numFrames
    for i = 1:size(featDesc,1)
        ulCornerCol = max(colLocations(i,k-1) - 5,1);
        ulCornerRow = max(rowLocations(i,k-1) - 5,1);

        for row = 1:maxVertDisp
           for col = 1:maxHorizDisp
               rowStart = ulCornerRow+row;
               rowEnd = ulCornerRow+row+7;
               colStart = ulCornerCol+col;
               colEnd = ulCornerCol+col+7;
               if(rowEnd <= size(framesTrimmed,1) && colEnd <= size(framesTrimmed,2))
                   patch = framesTrimmed(rowStart:rowEnd, ...
                   colStart:colEnd,k);
                   patchVec = patch(:);
                   SSDs(row,col) = sum((featDesc(i,:,k-1) - patchVec').^2);
               else
                   SSDs(row,col) = NaN;
               end
           end
        end

        minSSD = min(SSDs(:));
        if(~isnan(minSSD))
            [r,c] = find(SSDs <= minSSD);

            % getting the location of the center of the best patch within our box
            rowLocations(i,k) = ulCornerRow+r(1)+3;
            colLocations(i,k) = ulCornerCol+c(1)+3;

            % save best patch to feature descriptors matrix
            p = framesTrimmed((ulCornerRow+r):(ulCornerRow+r+7), ...
                (ulCornerCol+c):(ulCornerCol+c+7),k);
            featDesc(i,:,k) = p(:);
        else
            rowLocations(i,k) = NaN;
            colLocations(i,k) = NaN;
            featDesc(i,:,k) = NaN;
        end
    end
end


%%
% Visualization of movement
figure;
% for i=1:numKeypoints
%     plot(colLocations(i,:),rowLocations(i,:));
%     hold on;
% end

plot(colLocations(83,:),rowLocations(83,:));
%%
figure;
for i=1:numKeypoints
    plot(rowLocations(i,:));
    hold on;
end

%%
figure;
imshow(framesTrimmed(:,:,1),[]);
hold on;

for i = 1:size(find(kps(:,:,1) > 0))
    plot(c1(i), r1(i), 'r+');
end
hold on;

for i=1:numKeypoints
    plot(rowLocations(i,:));
    hold on;
end

%%

% discard the least stable keypoints
pos2 = zeros(size(colLocations,1),size(colLocations,2) - 1);
pos2(:,:) = colLocations(:,2:end);
amtMoved = abs(colLocations(:,1:end - 1) - pos2);
maxAmtMoved = max(amtMoved,[],2);

maxAmtMovedFiltered = maxAmtMoved(~isnan(maxAmtMoved));

medianMaxAmtMoved = median(maxAmtMovedFiltered);

%% Citations
% Krishnamurthy, R. Video Processing in Matlab. MathWorks, 2019,
% https://www.mathworks.com/videos/video-processing-in-matlab-68745.html



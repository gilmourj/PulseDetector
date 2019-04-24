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

%% Read in Video
vidReader = VideoReader('Headshot.mp4');
numFrames = 0;

% preallocate space to store frames
frames = zeros(1280,720,199);

while hasFrame(vidReader)
   numFrames = numFrames + 1;
   frame = imrotate(readFrame(vidReader),-90);
   frame = rgb2gray(frame);
   frames(:,:,numFrames) = frame;
end

frameHeight = size(frames,1);
frameWidth = size(frames,2);

%% Process Video

% preallocate space for new smaller frames
%trimmedFrames = zeros(311,227,numFrames);
framesTrimmed = zeros(311,227,numFrames);

for frame = 1:numFrames
    % get rid of extra space around face and store in new array
    framesTrimmed(:,:,frame) = frames(484:794,225:451,frame);
    
    % set area around eyes to NaN
    framesTrimmed(54:140,:,frame) = NaN;
end

%% Detect Features
% preallocate space for feature detections
kps = zeros(311,227,numFrames);

for frame = 1:numFrames
    kps(:,:,frame) = kpdet1(framesTrimmed(:,:,frame));
end

% find indices of keypoints
[r1, c1] = find(kps(:,:,1) > 0);

figure;
imshow(framesTrimmed(:,:,1),[]);
hold on;

for i = 1:size(find(kps(:,:,1) > 0))
    plot(c1(i), r1(i), 'r+');
end

%% Match Features

% get feature descriptors for first frame

for i = 1:(size(r1) - 1)
    
end

featDesc1 = kpfeat(frames(:,:,1),kps(:,:,1));




%% Citations
% Krishnamurthy, R. Video Processing in Matlab. MathWorks, 2019,
% https://www.mathworks.com/videos/video-processing-in-matlab-68745.html



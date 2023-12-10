%Training Code

clc;
clear all;


positiveImagesFolder = 'C:\Users\Usuario\OneDrive - Hanzehogeschool Groningen\Escritorio\Matlab Uni\CV-Project\Project CV\matlab-viola-jones\trainHaar\positive';
negativeImagesFolder = 'C:\Users\Usuario\OneDrive - Hanzehogeschool Groningen\Escritorio\Matlab Uni\CV-Project\Project CV\matlab-viola-jones\trainHaar\negative';

%positiveSize = numel(dir(positiveImagesFolder))-3;
%negativeSize = numel(dir(negativeImagesFolder))-2;
positiveSize = 20;
negativeSize = 24;

%%%Conversion to Integral Image%%%
%Initialize image array
Licenses = cell(1,positiveSize);
nonLicenses = cell(1,negativeSize);

%Iteration through every image with license plate
fprintf('License Images Reading...\n');
for LicenseNum = 1:positiveSize

    str = 'positive/';
    img = sprintf('Cars_%d',LicenseNum);
    fullPath = strcat(str,img,'.jpg');
    disp(fullPath)
    if ~exist(fullPath, 'file')
        continue; % Skip to the next iteration if the image does not exist
    end
    img = imread(fullPath);
    img = im2gray(img);
    % convert to integral image
    integral = integralImg(img);
    % append to image array
    Licenses{LicenseNum} = integral;
end
allImages = Licenses;

% iterate through each non-License image to get corresponding integral images
fprintf('Reading Non-License Images\n');
for nonLicenseNum = 1:negativeSize 
    % read non-license image
    str = 'negative/';
    img = sprintf('Cars_%d',nonLicenseNum);
    fullPath = strcat(str,img,'.jpg');
    disp(fullPath)
    % Check if the image exists
    if ~exist(fullPath, 'file')
        continue; % Skip to the next iteration if the image does not exist
    end

    img = imread(fullPath);
    %Assure is in gray scale
    img = im2gray(img);
    % convert to integral image
    integral = integralImg(img);
    % append to image array
    nonLicenses{nonLicenseNum} = integral;
    % append to full list of images
    allImages{nonLicenseNum+positiveSize} = integral;
end

%%% Construction of Haar Feature %%%

    % haar = the haar-like feature type
    % dimX, dimY = the x,y dimensions of the original haar features
    % pixelX, pixelY = the x,y index value for the starting pixel of
    % each haar feature
    % haarX, haarY = the x,y dimensions of the transformed haar features
    % Haar feature dimensions
% Haar feature dimensions
haars = [1,2; 2,1; 1,3; 3,1; 2,2];

% Image dimensions
imageWidth = 47;
imageHeight = 17;

% Initialize total number of features
numHaarFeatures = 0;

% Calculate the number of features for each Haar type and scale
for i = 1:size(haars, 1)
    baseHaarWidth = haars(i, 1);
    baseHaarHeight = haars(i, 2);
    
    % Iterate over possible scales
    for scaleWidth = 1:((imageWidth-1) / baseHaarWidth)
        for scaleHeight = 1:((imageHeight -1) / baseHaarHeight)
            haarWidth = baseHaarWidth * scaleWidth;
            haarHeight = baseHaarHeight * scaleHeight;
            
            % Number of positions where this scaled Haar feature can fit in the image
            positionsX = (imageWidth - 1) - haarWidth + 1;
            positionsY = (imageHeight - 1) - haarHeight + 1;

            % Total number of features of this type and scale
            numFeaturesThisTypeAndScale = positionsX * positionsY;

            % Add to total number of Haar features
            numHaarFeatures = numHaarFeatures + numFeaturesThisTypeAndScale;
        end
    end
end

% Output total number of Haar features
fprintf('Total number of Haar features considering scales: %d\n', numHaarFeatures);

totalImages = positiveSize + negativeSize;

% Initialize the datafeatures matrix
datafeatures = zeros(totalImages, numHaarFeatures);

% Initialize the dataclass array
dataclass = [ones(1, positiveSize), -ones(1, negativeSize)];

% Initialize the mapping structure
haarFeatureMapping = struct('haarType', {}, 'dimX', {}, 'dimY', {}, 'pixelX', {}, 'pixelY', {});

% Loop over each image
for imgIndex = 1:totalImages
    currentImage = allImages{imgIndex};
    integralImage = currentImage; % Assuming this is already an integral image

    % Initialize a feature index for each image
    featureIndex = 1;

    % Iterate over Haar types and scales
    for haarType = 1:size(haars, 1)
        baseDimX = haars(haarType, 1);
        baseDimY = haars(haarType, 2);

        % Iterate over scales
        for scaleX = 1:(imageWidth / baseDimX)
            for scaleY = 1:(imageHeight / baseDimY)
                dimX = baseDimX * scaleX;
                dimY = baseDimY * scaleY;

                % Iterate over positions within the image for the scaled Haar feature
                for pixelX = 1:(imageWidth - dimX )
                    for pixelY = 1:(imageHeight - dimY )
                        % Calculate Haar feature value for the scaled feature
                        haarValue = calcHaarValues(integralImage, haarType, pixelX, pixelY, dimX, dimY);
                        fprintf('Haar Type: %d, Value: %f, dimX:%d, dimY:%d, pixelX:%d, pixelY:%d\n', haarType, haarValue, dimX, dimY, pixelX, pixelY)
                        
                        % Store in datafeatures matrix
                        datafeatures(imgIndex, featureIndex) = haarValue;

                        % Store the corresponding Haar feature attributes in the mapping
                        haarFeatureMapping(featureIndex) = struct('haarType', haarType, 'dimX', dimX, 'dimY', dimY, 'pixelX', pixelX, 'pixelY', pixelY);

                        % Update featureIndex
                        featureIndex = featureIndex + 1;
                    end
                end
            end
        end
    end
end

% Create a dynamic filename based on positiveSize and negativeSize for
% easier use for trainning part
filename = sprintf('dataFeaturesClass_Pos%d_Neg%d.mat', positiveSize, negativeSize);

% Save the datafeatures and dataclass variables to the dynamically named .mat file
save(filename, 'datafeatures', 'dataclass',"haarFeatureMapping");

%% 
%Training phase, trying to implement cascade mechanism 
%Load dafeatures sets
load("dataFeaturesClass_Pos400_Neg450.mat","datafeatures","dataclass","haarFeatureMapping");
load("dataFeaturesTestClass_Pos20_Neg30.mat","datafeaturesTest","dataclassTest");
datafeaturesTest = datafeaturesTest(end-(49):end, :);

% Number of iterations for each AdaBoost model

itt = 25;

% Training each stage
numberOfStages = 4;
models = cell(1, numberOfStages);

for stage = 1:numberOfStages
    fprintf('Training stage %d\n', stage);
    [model, falsePositives] = adabost(datafeaturesTest,dataclassTest,datafeatures, dataclass, itt);
    models{stage} = model;
    disp(size(falsePositives))
    
    % If there are more stages, add false positives to the training set
    if stage < numberOfStages
        % Update negative examples with false positives
        numFalsePositives = size(falsePositives, 1);
        datafeatures = [datafeatures; falsePositives]; % Add false positives to features
        dataclass = [dataclass, -ones(1, numFalsePositives)]; % Add negative labels for false positives
    end
end

filename = sprintf('model_itt_%d_nOfStages_%d_size_%d.mat', itt, numberOfStages,size(datafeatures(:,1)));

% Save the datafeatures and dataclass variables to the dynamically named .mat file
save(filename, "models");



%% 
clc;
load("model_itt_25_nOfStages_4_size_860.mat");
load("dataFeaturesClass_Pos400_Neg450.mat","datafeatures","dataclass","haarFeatureMapping");

%Sliding window approach to find 
inputImagePath = "C:\Users\Usuario\OneDrive - Hanzehogeschool Groningen\Escritorio\Matlab Uni\CV-Project\Project CV\001\CroppedVehicles\00430.jpg_vehicle_2.jpg";
if isfile(inputImagePath)
    % Load the original image
    originalImage = imread(inputImagePath);
end

matches = [];
originalImage = im2gray(originalImage); 
baseWindowSize = [17, 47]; 
baseStepSize = 5; % Base step size for scale 1
scales = 1; %Only using scale 1
pyramidItt = 8; %Number of pyramid stages

model = cell(7, 1); % Create a cell array with 7 rows and 1 column

% Assign each stage to a cell in the model array
model{1} = selectedClassifiers(1:2,:);
model{2} = selectedClassifiers(3:9,:);
model{3} = selectedClassifiers(10:30,:);
model{4} = selectedClassifiers(31:74,:);   
model{5} = selectedClassifiers(75:124,:);
model{6} = selectedClassifiers(125:199,:);
model{7} = selectedClassifiers(200:300,:);


for itr = 1:pyramidItt
    printout = strcat('Iteration #',int2str(itr),'\n');
    fprintf(printout);
    for scale = scales
        % Rescale image
        inputImage = imresize(originalImage, scale);
        
        % Adjust window size and step size based on scale
        windowSize = round(baseWindowSize * scale);
        stepSize = round(baseStepSize * scale);
    
        maxX = size(inputImage, 2) - windowSize(2) + 1;
        maxY = size(inputImage, 1) - windowSize(1) + 1;
    
        for x = 1:stepSize:maxX
            for y = 1:stepSize:maxY
                window = inputImage(y:y+windowSize(1)-1, x:x+windowSize(2)-1);
                normalizedWindow = imresize(window, baseWindowSize);
                normalizedWindow = integralImage(normalizedWindow);
               
                isPositive = true;
                for stage = 1:numberOfStages
                    classificationScore = cascadeClassifier(model{stage}, normalizedWindow ,haarFeatureMapping);
                    if classificationScore ~= 1
                        isPositive = false;
                        break; % Window rejected by this stage
                    end
                end
    
                if isPositive
                    % Window passed all stages, mark as positive detection
                    fprintf("Possible match in steps x:%d & y:%d in pyramid itt %d",x,y,itr);
                    matches = [matches; x, y, classificationScore,scale];
                end
            end
        end
    end
    % create next image pyramid level
    tempImg = imresize(inputImage,.8);
    img = tempImg;
    %new input image
    inputImage = integralImg(img);
end

%% 

%Displaying possible areas of interest
imshow(inputImage);
hold on; % This keeps the image displayed while we draw the rectangle

% Assuming coords is a 2-element vector [x, y]

x = matches(1);
y = matches(2);

% Define the width and height of the rectangle

% Draw the rectangle
for i = 1:size(matches, 1)
    width = 47;
    height = 17;
    x = matches(i, 1);
    y = matches(i, 2);
    scale = matches(i,4);
    width = round(width*scale);
    height = round(height*scale);

    rectangle('Position', [x, y, width, height], 'EdgeColor', 'r', 'LineWidth', 2);
end

hold off; % Release the hold on the figure



%% 
%%%Code that obtains the haarFeatures of one image
function [datafeatures] = getHaarFeatures(inputImage)
    % Read and convert the image to an integral image

    integralImage = integralImg(inputImage);
  
    
    % Haar feature dimensions
    haars = [1,2; 2,1; 1,3; 3,1; 2,2];
    
    % Image dimensions (adjust these according to your specific image)
    imageWidth = size(integralImage, 2);
    imageHeight = size(integralImage, 1);
    
    % Initialize the total number of features
    numHaarFeatures = 3433;
    
    % Output the total number of Haar features
    fprintf('Total number of Haar features considering scales: %d\n', numHaarFeatures);
    
    % Initialize the datafeatures array for the single image
    datafeatures = zeros(1, numHaarFeatures);
    
    % Initialize feature index
    featureIndex = 1;
    
    % Iterate over Haar types
    for haarType = 1:size(haars, 1)
        dimX = haars(haarType, 1);
        dimY = haars(haarType, 2);
    
        % Iterate over positions and scales within the image
        for pixelX = 1:(imageWidth - dimX - 1)
            for pixelY = 1:(imageHeight - dimY - 1)
                % Calculate Haar feature value
                haarValue = calcHaarValues(integralImage, haarType, pixelX, pixelY, dimX, dimY);
                fprintf('Haar Type: %d, Value: %f, dimX:%d, dimY:%d, pixelX:%d, pixelY:%d\n', haarType, haarValue, dimX, dimY, pixelX, pixelY);
                
                % Store in datafeatures array
                datafeatures(featureIndex) = haarValue;
    
                % Update featureIndex
                featureIndex = featureIndex + 1;
            end
        end
    end
end

% Now datafeatures contains the Haar features for the single image


function outimg = integralImg (inimg)
    % cumulative sum for each pixel of all rows and columns to the left and
    % above the corresponding pixel
    outimg = cumsum(cumsum(double(inimg),2));
end

function intensity = getCorners(img,startX,startY,endX,endY)
    a = img(startY,startX);
    b = img(startY,endX);
    c = img(endY,startX);
    d = img(endY,endX);
    intensity = d-(b+c)+a; % by property of the integral image
end

function estimateclass = cascadeClassifier(model, inputImage, haarFeatureMapping)
    % Preprocess the input image (this should match your training preprocessing)
    % For example, if you used integral images:
    % processedImage = integralImage(inputImage);
    inputImage = integralImg(inputImage);

    
    % Initialize the sum of weak classifier results
    estimateclasssum = 0;

    % Loop through each weak classifier in the model
    for t = 1:length(model)
        % Extract the relevant features for this classifier
        % This is highly dependent on how your features are defined
        % For example, if your features are Haar-like:
        
        %Retrieve values based on the haarFeatureMapping 
        haarTypeValue = haarFeatureMapping(model(t).dimension).haarType;
        dimXValue = haarFeatureMapping(model(t).dimension).dimX;
        dimYValue = haarFeatureMapping(model(t).dimension).dimY;
        pixelXValue = haarFeatureMapping(model(t).dimension).pixelX;
        pixelYValue = haarFeatureMapping(model(t).dimension).pixelY;

        featureValue = calcHaarValues(inputImage, haarTypeValue, dimXValue, dimYValue, pixelXValue,pixelYValue);
        
        % Apply the weak classifier threshold
        if model(t).direction == 1
            weakResult = double(featureValue >= model(t).threshold);
        else
            weakResult = double(featureValue < model(t).threshold);
        end
        weakResult(weakResult == 0) = -1;

        % Add to the sum of weak classifier results, weighted by alpha
        estimateclasssum = estimateclasssum + model(t).alpha * weakResult;
    end

    % Final classification decision
    estimateclass = sign(estimateclasssum);
end


function haarValue = calcHaarValues(img, haarType, pixelX, pixelY, dimX, dimY)
    % integralImage: The integral image
    % haarType: Type of Haar feature (determined by its dimensions)
    % pixelX, pixelY: Top left coordinates of the Haar feature
    % dimX, dimY: Dimensions of the Haar feature

    % Initialize the Haar value
    haarValue = 0;

    function sum = rectSum(x1, y1, x2, y2)
        sum = img(y2, x2) - img(y1, x2) - img(y2, x1) + img(y1, x1);
    end


     switch haarType
        case 1 % Horizontal two-rectangle feature
            midX = pixelX + round(dimX/2);
            leftSum = rectSum(pixelX, pixelY, midX, pixelY + dimY);
            rightSum = rectSum(midX, pixelY, pixelX + dimX, pixelY + dimY);
            haarValue = leftSum - rightSum;  
        case 2 % Vertical two-rectangle feature
            midY = pixelY + round(dimY/2);
            topSum = rectSum(pixelX, pixelY, pixelX + dimX, midY);
            bottomSum = rectSum(pixelX, midY, pixelX + dimX, pixelY + dimY);
            haarValue = topSum - bottomSum;

        case 3 % Horizontal three-rectangle feature
            thirdX = round(dimX / 3);
            leftSum = rectSum(pixelX, pixelY, pixelX + thirdX, pixelY + dimY);
            middleSum = rectSum(pixelX + thirdX, pixelY, pixelX + 2*thirdX, pixelY + dimY);
            rightSum = rectSum(pixelX + 2*thirdX, pixelY, pixelX + dimX, pixelY + dimY);
            haarValue = leftSum - middleSum + rightSum;

        case 4 % Vertical three-rectangle feature
            thirdY = round(dimY / 3);
            topSum = rectSum(pixelX, pixelY, pixelX + dimX, pixelY + thirdY);
            middleSum = rectSum(pixelX, pixelY + thirdY, pixelX + dimX, pixelY + 2*thirdY);
            bottomSum = rectSum(pixelX, pixelY + 2*thirdY, pixelX + dimX, pixelY + dimY);
            haarValue = topSum - middleSum + bottomSum;

        case 5 % Four-rectangle feature
            midX = pixelX + round(dimX/2);
            midY = pixelY + round(dimY/2);
            topLeftSum = rectSum(pixelX, pixelY, midX, midY);
            topRightSum = rectSum(midX, pixelY, pixelX + dimX, midY);
            bottomLeftSum = rectSum(pixelX, midY, midX, pixelY + dimY);
            bottomRightSum = rectSum(midX, midY, pixelX + dimX, pixelY + dimY);
            haarValue = topLeftSum - topRightSum - bottomLeftSum + bottomRightSum;
    end
end

function [newWeights,alpha] = adaboost(classifier, images, imgWeights,positiveSize,imgsSize)
captures = zeros(imgsSize,1);
error = 0;

for i = 1:imgsSize
    img = images{i};
    % obtains classifier metadata from fields in the row vector 
    haar = classifier(1);
    pixelX = classifier(2);
    pixelY = classifier(3);
    haarX = classifier(4);
    haarY = classifier(5);
    % calculates intensity difference between black-white region of the
    % Haar feature and checks against the precalculated range
    haarVal = calcHaarValues(img,haar,pixelX,pixelY,haarX,haarY);
    if haarVal >= classifier(9) && haarVal <= classifier(10) % if falls between correct value
        if i <= positiveSize % if its a license plate
            captures(i) = 1; % correct capture
        else
            captures(i) = 0; % error
            error = error + imgWeights(i); % increase weighted error count
        end
    else % if falls outside the expected range
        if i <= positiveSize % if is a Plate
            captures(i) = 0;
            error = error + imgWeights(i); % error
        else 
            captures(i) = 1;
        end
    end
end

alpha = 0.5*log((1-error)/error); % updates classifier weight (alpha)

% modifies images' weights by whether it is a successful capture or not
% correct captures result in lower weights; false captures result in higher
% weight to put more emphasis on them
for i = 1:imgsSize
    if captures(i) == 0
        imgWeights(i) = imgWeights(i).*exp(alpha);
    else
        imgWeights(i) = imgWeights(i).*exp(-alpha);
    end
end
imgWeights = imgWeights./sum(imgWeights); % normalize image weights
newWeights = imgWeights; % pass as function output
end

%This code generates the datafeatesTest, used in the cascade classifier to
%enhance the object detection
clear all;
clc; 
positiveImagesFolder = '..\testpostives'
negativeImagesFolder = '..\testnegatives';

%Image selection
displacement = 0;

positiveSize = 4;
negativeSize = 4;

%%%Conversion to Integral Image%%%
%Initialize image array
Licenses = cell(1,positiveSize);
nonLicenses = cell(1,negativeSize);

%Iteration through every image with license plate
fprintf('License Images Reading...\n');
for LicenseNum = 1:positiveSize+displacement

    str = 'testpositive/';
    img = sprintf('Cars_%d',LicenseNum);
    fullPath = strcat(str,img,'.jpg');
    img = imread(fullPath);
    % convert to integral image
    integral = integralImg(img);
    % append to image array
    Licenses{LicenseNum} = integral;
end
allImages = Licenses;

% iterate through each non-License image to get corresponding integral images
fprintf('Reading Non-License Images\n');
for nonLicenseNum = 1:negativeSize+displacement 
    % read non-license image
    str = 'testnegatives/';
    img = sprintf('Cars_%d',nonLicenseNum);
    fullPath = strcat(str,img,'.jpg');
    disp(fullPath)
    % Check if the image exists
    if ~exist(fullPath, 'file')
        continue; % Skip to the next iteration if the image does not exist
    end

    img = imread(fullPath);
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
datafeaturesTest = zeros(totalImages, numHaarFeatures);

% Initialize the dataclass array
dataclassTest = [ones(1, positiveSize), -ones(1, negativeSize)];

% Initialize the mapping structure
haarFeatureMapping = struct('haarType', {}, 'dimX', {}, 'dimY', {}, 'pixelX', {}, 'pixelY', {});

% Loop over each image
for imgIndex = 1:totalImages+displacement
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
                        datafeaturesTest(imgIndex, featureIndex) = haarValue;

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

datafeaturesTest = datafeaturesTest(end-(totalImages-1):end, :);
% Create a dynamic filename based on positiveSize and negativeSize for
% easier use for trainning part
filename = sprintf('dataFeaturesTestClass_Pos%d_Neg%d.mat', positiveSize, negativeSize);

% Save the datafeatures and dataclass variables to the dynamically named .mat file
save(filename, 'datafeaturesTest', 'dataclassTest');

%% 


function haarValue = calcHaarValues(integralImage, haarType, pixelX, pixelY, dimX, dimY)
    % integralImage: The integral image
    % haarType: Type of Haar feature (determined by its dimensions)
    % pixelX, pixelY: Top left coordinates of the Haar feature
    % dimX, dimY: Dimensions of the Haar feature

    % Initialize the Haar value
    haarValue = 0;

    function sum = rectSum(x1, y1, x2, y2)
        sum = integralImage(y2, x2) - integralImage(y1, x2) - integralImage(y2, x1) + integralImage(y1, x1);
    end

     switch haarType
        case 1 % Horizontal two-rectangle feature
            midY = pixelY + round(dimY/2);
            topSum = rectSum(pixelX, pixelY, pixelX + dimX, midY);
            bottomSum = rectSum(pixelX, midY, pixelX + dimX, pixelY + dimY);
            haarValue = topSum - bottomSum;

        case 2 % Vertical two-rectangle feature
            midX = pixelX + round(dimX/2);
            leftSum = rectSum(pixelX, pixelY, midX, pixelY + dimY);
            rightSum = rectSum(midX, pixelY, pixelX + dimX, pixelY + dimY);
            haarValue = leftSum - rightSum;

        case 3 % Horizontal three-rectangle feature
            thirdY = round(dimY / 3);
            topSum = rectSum(pixelX, pixelY, pixelX + dimX, pixelY + thirdY);
            middleSum = rectSum(pixelX, pixelY + thirdY, pixelX + dimX, pixelY + 2*thirdY);
            bottomSum = rectSum(pixelX, pixelY + 2*thirdY, pixelX + dimX, pixelY + dimY);
            haarValue = topSum - 2*middleSum + bottomSum;

        case 4 % Vertical three-rectangle feature
            thirdX = round(dimX / 3);
            leftSum = rectSum(pixelX, pixelY, pixelX + thirdX, pixelY + dimY);
            middleSum = rectSum(pixelX + thirdX, pixelY, pixelX + 2*thirdX, pixelY + dimY);
            rightSum = rectSum(pixelX + 2*thirdX, pixelY, pixelX + dimX, pixelY + dimY);
            haarValue = leftSum - 2*middleSum + rightSum;

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

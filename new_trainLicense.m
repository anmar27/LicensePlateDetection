%Training Code

clc;
clear all;


positiveImagesFolder = '../positiveFolder';
negativeImagesFolder = '../negativeFolder';

%positiveSize = numel(dir(positiveImagesFolder))-3;
%negativeSize = numel(dir(negativeImagesFolder))-2;
positiveSize = 600;
negativeSize = 850;
imgsSize = 600+850;

%%%Conversion to Integral Image%%%
%Initialize image array
Licenses = cell(1,positiveSize);
nonLicenses = cell(1,negativeSize);

%Iteration through every image with license plate
fprintf('License Images Reading...\n');
for LicenseNum = 1:positiveSize

    str = 'positive/';
    img = sprintf('Cars%d',LicenseNum);
    fullPath = strcat(str,img,'.png');
    disp(fullPath)
    if ~exist(fullPath, 'file')
        continue; % Skip to the next iteration if the image does not exist
    end
    img = imread(fullPath);
    img = im2gray(img);
    % convert to integral image
    integralImgage = integralImgage(img);
    % append to image array
    Licenses{LicenseNum} = integralImgage;
end
allImages = Licenses;

% iterate through each non-License image to get corresponding integral images
fprintf('Reading Non-License Images\n');
for nonLicenseNum = 1:negativeSize 
    % read non-license image
    str = 'negative/';
    img = sprintf('random_Cars%d',nonLicenseNum);
    fullPath = strcat(str,img,'.png');
    disp(fullPath)
    % Check if the image exists
    if ~exist(fullPath, 'file')
        continue; % Skip to the next iteration if the image does not exist
    end

    img = imread(fullPath);
    %Assure is in gray scale
    img = im2gray(img);
    % convert to integral image
    integralImgage = integralImgage(img);
    % append to image array
    nonLicenses{nonLicenseNum} = integralImgage;
    % append to full list of images
    allImages{nonLicenseNum+positiveSize} = integralImgage;
end

%Construction of Haar Features
%%% Variable Definitions %%%
    % haar = the haar-like feature type
    % dimX, dimY = the x,y dimensions of the original haar features
    % pixelX, pixelY = the x,y index value for the starting pixel of
    % each haar feature
    % haarX, haarY = the x,y dimensions of the transformed haar features

fprintf('Constructing Haar Features\n');
% initialize image weights
imgWeights = ones(positiveSize+negativeSize,1)./(positiveSize+negativeSize);
% matrix of haar feature dimensions
haars = [1,2;2,1;1,3;3,1;2,2];
% size of training images
windowY = 17;
windowX = 47;

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

%number of training itteration
itt =  3;

%number of training iterations
for iterations = 1:itt
    % initialize classifier container
    weakClassifiers = {};
    % iterate through features
    for haarType = 1:5
        printout = strcat('Working on Haar #',int2str(haarType),'\n');
        fprintf(printout);
        % get x dimension
        dimX = haars(haarType,1);
        % get y dimension
        dimY = haars(haarType,2);
        % iterate through available pixels in window
        for pixelX = 2:windowX-dimX
            for pixelY = 2:windowY-dimY
                % iterate through possible haar dimensions for pixel
                for haarX = dimX:dimX:windowX-pixelX
                    for haarY = dimY:dimY:windowY-pixelY
                        %Initalization of vectors of Haar Features for both

                         haarVector_NonLicenses = zeros(1,negativeSize);
                         haarVector_Licenses = zeros(1,positiveSize);
                         % Iterate through each integral image in License array
                         for img = 1:positiveSize
                             % calculate resulting feature value for each image
                             value = calcHaarValues(Licenses{img},haarType,pixelX,pixelY,haarX,haarY);
                             % store feature value
                            haarVector_Licenses(img) = value;
                         end
                        
                          % Iterate through each integral image in the nonLicense array
                          for img = 1:negativeSize
                             % calculate resulting feature value for each image
                             value = calcHaarValues(nonLicenses{img},haarType,pixelX,pixelY,haarX,haarY);
                             % store feature value
                            haarVector_NonLicenses(img) = value;
                         end
                        
                          % Create variables with distribution values of haar features in positive & negative images
                          LicenseMax = max(haarVector_Licenses);
                          LicenseMin = min(haarVector_Licenses);
                          LicenseMean = mean(haarVector_Licenses);                                          
                          NonLicenseMean = mean(haarVector_NonLicenses);
                        
                          %Discard in case of means too  close
                         meanDiffThreshold = 5; %Possibly needed to be adjusted depending on the data set
                        meanDifference = abs(LicenseMean - NonLicenseMean);
                        
                        if meanDifference < meanDiffThreshold
                            % Skip this feature as it is not discriminative enough
                            continue;
                        end
                        % Selected 100 steps for finding the threshold
                        % iteratively
                        for iter = 1:100
                          %Initialize variable used for marking correct
                         %or incorrect classification
                            featureClassification = ones(size(imgWeights,1),1); % 0 = correctly classified | 1 = missclasiffied
                            %Define min & max rating based on mean and max
                            %and minimun global values
                            minRating = LicenseMean-abs((iter/100)*(LicenseMean-LicenseMin));
                            maxRating = LicenseMean+abs((iter/100)*(LicenseMax-LicenseMean));
                            % capture all true positives values
                            for value = 1:positiveSize
                                if haarVector_Licenses(value) >= minRating && haarVector_Licenses(value) <= maxRating
                                  featureClassification(value) = 0; %Correctly classified
                                end
                            end
                            % Sum of weights incorrectly classified 
                            LicenseRating = sum(imgWeights(1:positiveSize).*featureClassification(1:positiveSize));
                            if LicenseRating < 0.05 % if less than 5% of License plates are misclassified
                                % capture all false positive values
                                for value = 1:negativeSize
                                if haarVector_NonLicenses(value) >= minRating && haarVector_NonLicenses(value) <= maxRating
                                else
                                    featureClassification(value+positiveSize) = 0; % Correctly clasified
                                end
                           end
                            % Sum of all weights of those nonLicense
                                                        
                            nonLicenseRating = sum(imgWeights(positiveSize+1:negativeSize+positiveSize).*featureClassification(positiveSize+1:negativeSize+positiveSize));
                            % total error
                            totalError = sum(imgWeights.*featureClassification);
                                if totalError < .5 % When total error is lower than the half (Better than random)%
                                    % Store weak classifier
                                    Counter = Counter+1;
                                    RatingDiff = [RatingDiff,(1-LicenseRating)-nonLicenseRating];
                                    LicenseRating = [LicenseRating,1-LicenseRating];
                                    NonLicenseRating = [NonLicenseRating,nonLicenseRating];
                                    TotalError = [TotalError,totalError];
                                    LowerBound = [LowerBound,minRating];
                                    UpperBound = [UpperBound,maxRating];
                                end
                            end
                        end

                        % In case of potential features find index of one with the 
                        % maximum difference between true and false positives
                           if size(RatingDiff) > 0
                                maxRatingIndex = -inf; % Putting -infinite value to make sure of selecting a correct idex
                                maxRatingDiff = max(RatingDiff);
                                 for index = 1:size(RatingDiff,2)
                                    if RatingDiff(index) == maxRatingDiff
                                        maxRatingIndex = index; % found the index of maxRatingDiff
                                        break;
                                    end
                                end
                            end


                        % Store classifier related data into Classifier
                        % variable

                        if size(storeRatingDiff) > 0
                           Classifier = [haarType,pixelX,pixelY,haarX,haarY,...
                                maxRatingDiff,storeLicenseRating(maxRatingIndex),storeNonLicenseRating(maxRatingIndex),...
                                storeLowerBound(maxRatingIndex),storeUpperBound(maxRatingIndex),...
                                storeTotalError(maxRatingIndex)];

                            % Adaboost for updating the weights and obtain
                            % alpha value of feature
                            [imgWeights,alpha] = adaboost(Classifier,allImages,imgWeights,positiveSize,imgsSize);
                            % append alpha to classifier metadata
                            Classifier = [Classifier,alpha];
                            %Store the corresponding classifier in
                            %weakClassifier array
                            weakClassifiers{size(weakClassifiers,2)+1} = Classifier;          
                            end
                        end
                    end
                end
            end
        end
        printout = strcat('Finished Haar #',int2str(haarType),'\n');
        fprintf(printout);
    end 
end
%% 

%Sorting the weak classifiers based on alpha values
fprintf('Make strong classifiers from sorting according to alpha values\n');
alphas = zeros(size(weakClassifiers,2),1);
for i = 1:size(alphas,1)
    % extract alpha column from classifier metadata
    alphas(i) = weakClassifiers{i}(12);
end

% sort weakClassifiers
tempClassifiers = zeros(size(alphas,1),2); % 2 column
% first column is simply original alphas
tempClassifiers(:,1) = alphas;
for i = 1:size(alphas,1)
    % second column is the initial index of alpha values wrt original alphas
   tempClassifiers(i,2) = i; 
end

tempClassifiers = sortrows(tempClassifiers,-1); % sort descending order

% number of strong classifiers tailored to our implementation, might vary
selectedClassifiers = zeros(size(weakClassifiers,2),12);
for i = 1:size(weakClassifiers,2)
    selectedClassifiers(i,:) = weakClassifiers{tempClassifiers(i,2)};
end

% save final set of strong classifiers into a .mat file for easier access
save('finalClassifiers.mat','selectedClassifiers');
%% 
%Detection Part Based on final classifier model

Licenses,LicensesBounds = detectLicenses('C:\Users\Usuario\OneDrive - Hanzehogeschool Groningen\Escritorio\Matlab Uni\CV-Project\Project CV\002\CroppedVehicles\002_04320.jpg_vehicle_5.jpg')

%% 

function [Licenses,licencesBound] = detectLicenses(img)
% preprocessing by Gaussian filtering
img2 = img; % keep a copy of the original color 3D image
img = imread(img);
img = rgb2gray(img);
img = conv2(img,fspecial('gaussian',3,3),'same');
      
% get image parameters
[m,n] = size(img);

% number of iterations it will directly affect the possible matches
pyramidItt = 10; 
Licenses = []; % empty by default
%Number of matches
num_top_matches = 5;
Licenses = zeros(num_top_matches, 5); % Store top 5 matches with iteration index


% compute integral image
intImg = integralImg(img);

% load finalClassifiers
load trainedClassifiers.mat % 200 classifiers

%Cascaded Detector Structure: 7 levels, 300 classifiers

stage1 = selectedClassifiers(1:2,:);
stage2 = selectedClassifiers(3:9,:);
stage3 = selectedClassifiers(10:30,:);
stage4 = selectedClassifiers(31:74,:);
stage5 = selectedClassifiers(75:124,:);
stage6 = selectedClassifiers(125:199,:);
stage7 = selectedClassifiers(200:300,:);

% iterate through each window size/pyramid level
    for itt = 1:pyramidItt
        fprintf('Iteration #%d\n', itt);
        for i = 1:2:m-18
            if i + 18 > m 
                break; % boundary case check
            end
            for j = 1:2:n-47
                if j + 47 > n
                    break; % boundary case check
                end
                window = intImg(i:i+17, j:j+46); 
    
                stages = {stage1, stage2, ... ,stage7};
                thresholds = [1,0.5,0.5,0.5,0.6,0.6,0.6];
                if checkCascade(stages, window, thresholds)
                    % save rectangular corner coordinates
                    bounds = [j, i, j+46, i+17, itr];
                    fprintf('License Plate detected!\n');
                    Licenses = [Licenses; bounds];
                end
            end
        end
    
        % create next image pyramid level
        img = conv2(img,fspecial('gaussian',3,3),'same');
        img = imresize(img, 0.8);
        [m, n] = size(img);
        intImg = integralImg(img);
    end
    % create next image pyramid level
    tempImg = imresize(img,.8);
    img = tempImg;
    [m,n] = size(img);
    intImg = integralImg(img);
end

if size(Licenses,1) == 0 % No License detected
   error('No License detected! Try again with a larger value of scanItr.'); 
end

%Selecting best bound box
% upscale rectangular bound coordinates back to base level of pyramid
    licencesBound = zeros(size(Licenses,1),4);
    maxItr = max(Licenses(:,5)); % higher iterations have larger bounding boxes
    for i = 1:size(Licenses,1)
        licencesBound(i,:) = floor(Licenses(i,1:4)*1.2^(Licenses(i,5)-1));
    end
%filter out overlapping rectangular bounding boxes
startRow = 1;
for i = 1:size(LicenseBound,1)
   if LicenseBound(i,1) == 0
       startRow = startRow+1; % start with next row
  end
end
LicenseBound = LicenseBound(startRow:end,:); % get rid of 0-filled rows

% get the union of the areas of overlapping boxes
LicenseBound = [min(LicenseBound(:,1)),min(LicenseBound(:,2)),max(LicenseBound(:,3)),max(LicenseBound(:,4))];

figure,imshow(img2), hold on;
if(~isempty(licencesBound));
    for n=1:size(licencesBound,1)
        toleranceX = floor(0.1*(licencesBound(n,3)-licencesBound(n,1)));
        toleranceY = floor(0.1*(licencesBound(n,4)-licencesBound(n,2)));
        % original bounds
        x1=licencesBound(n,1); y1=licencesBound(n,2);
        x2=licencesBound(n,3); y2=licencesBound(n,4);
        % adjusted bounds to get wider License capture
        x1t=licencesBound(n,1)-toleranceX; y1t=licencesBound(n,2)-toleranceY;
        x2t=licencesBound(n,3)+toleranceX; y2t=licencesBound(n,4)+toleranceY;
        imSize = size(imread(img2));
        % if adjusted bounds will lead to out-of-bounds plotting, use original bounds
        if x1t < 1 || y1t < 1 || x2t > imSize(2) || y2t > imSize(1)
            fprintf('Out of bounds adjustments. Plotting original values...\n');
            plot([x1 x1 x2 x2 x1],[y1 y2 y2 y1 y1],'LineWidth',2);
        else
            plot([x1t x1t x2t x2t x1t],[y1t y2t y2t y1t y1t],'LineWidth',2);
        end
    end
end

title('Possible license matchs');
hold off;

end


function output = cascade(classifiers,img,thresh)
result = 0;
px = size(classifiers,1);
weightSum = sum(classifiers(:,12));
% iterate through each classifier
for i = 1:px
    classifier = classifiers(i,:);
    haar = classifier(1);
    pixelX = classifier(2);
    pixelY = classifier(3);
    haarX = classifier(4);
    haarY = classifier(5);
    % calculate the feature value for the subwindow using the current
    % classifier
    haarVal = calcHaarVal(img,haar,pixelX,pixelY,haarX,haarY);
    if haarVal >= classifier(9) && haarVal <= classifier(10)
        % increase score by the weight of the corresponding classifier
        score = classifier(12);
    else
        score = 0;
    end
   result = result + score;
end
% compare resulting weighted success rate to the threshold value
if result >= weightSum*thresh
    output = 1; % hit
else
    output = 0; % miss
end
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

% adaboost.m - boosts classifiers adaptively by updating their weights
% alpha values, and for individual images by updating image weights
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
    haarVal = calcHaarVal(img,haar,pixelX,pixelY,haarX,haarY);
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

function passed = checkCascade(stages, window, thresholds)
    for level = 1:length(stages)
        if cascade(stages{level}, window, thresholds(level)) ~= 1
            passed = false;
            return;
        end
        fprintf('Passed level %d cascade.\n', level);
    end
    passed = true;
end
%Training Code

clc;
clear all;


positiveImagesFolder = 'C:\Users\Usuario\OneDrive - Hanzehogeschool Groningen\Escritorio\Matlab Uni\CV-Project\Project CV\matlab-viola-jones\trainHaar\positiveImages1';
negativeImagesFolder = 'C:\Users\Usuario\OneDrive - Hanzehogeschool Groningen\Escritorio\Matlab Uni\CV-Project\Project CV\matlab-viola-jones\trainHaar\negativeimages1';

%positiveSize = numel(dir(positiveImagesFolder))-3;
%negativeSize = numel(dir(negativeImagesFolder))-2;
positiveSize = 15;
negativeSize = 30;

%%%Conversion to Integral Image%%%
%Initialize image array
Licenses = cell(1,positiveSize);
nonLicenses = cell(1,negativeSize);

%Iteration through every image with license plate
fprintf('License Images Reading...\n');
for LicenseNum = 1:positiveSize

    str = 'positiveImages1/';
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
fprintf('Reading Non-Face Images\n');
for nonLicenseNum = 1:negativeSize 
    % read non-license image
    str = 'negativeImages1/';
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

%Limiting to 3433 since we are not sonsidering resized features
%datafeatures = datafeatures(1:30,1:265144);

% Create a dynamic filename based on positiveSize and negativeSize for
% easier use for trainning part
filename = sprintf('dataFeaturesClass_Pos%d_Neg%d.mat', positiveSize, negativeSize);

% Save the datafeatures and dataclass variables to the dynamically named .mat file
save(filename, 'datafeatures', 'dataclass');

%% 
%Training phase, trying to implement cascade mechanism 
% Number of iterations for each AdaBoost model
itt = 10;

% Training each stage
numberOfStages = 3;
models = cell(1, numberOfStages);

for stage = 1:numberOfStages
    fprintf('Training stage %d\n', stage);
    
    [model, falsePositives] = trainCascadeStage(datafeatures, dataclass, itt);
    models{stage} = model;
    
    % If there are more stages, add false positives to the training set
    if stage < numberOfStages
        % Update negative examples with false positives
        numFalsePositives = size(falsePositives, 1);
        datafeatures = [datafeatures; falsePositives]; % Add false positives to features
        dataclass = [dataclass, -ones(1, numFalsePositives)]; % Add negative labels for false positives
    end
end

%% 

% Calculate the training accuracy
%accuracy = sum(dataclass == estimateclasstotal) / length(dataclass);
%fprintf('Training Accuracy: %.2f%%\n', accuracy * 100);

%% 

%Sliding window approach to find 
inputImagePath = "C:\Users\Usuario\OneDrive - Hanzehogeschool Groningen\Escritorio\Matlab Uni\CV-Project\Project CV\001\CroppedVehicles\00400.jpg_vehicle_3.jpg";
if isfile(inputImagePath)
    % Load the image
    inputImage = imread(inputImagePath);
end

%Transform into grayscale inputImage
inputImage = im2gray(inputImage);

matches = [];

windowSize = [17, 47]; 
stepSize = 5;

maxX = size(inputImage, 2) - windowSize(2) + 1;
maxY = size(inputImage, 1) - windowSize(1) + 1;

for x = 1:stepSize:maxX
    for y = 1:stepSize:maxY
        window = inputImage(y:y+windowSize(1)-1, x:x+windowSize(2)-1);
        %windowFeatures = getHaarFeatures(window);

        isPositive = true;
        for stage = 1:numberOfStages
            %classificationScore = adaboost('apply', windowFeatures, models{stage});
            classificationScore = applyAdaboostModelToImage(models{stage}, window ,haarFeatureMapping);
            if classificationScore ~= 1
                isPositive = false;
                break; % Window rejected by this stage
            end
        end

        if isPositive
            % Window passed all stages, mark as positive detection
            fprintf("Possible match in steps x:%d & y:%d",x,y);
            matches = [matches; x, y, classificationScore];
        end
    end
end


%% 

%Displaying possible areas of interest
imshow(inputImage);
hold on; % This keeps the image displayed while we draw the rectangle

% Assuming coords is a 2-element vector [x, y]

x = matches(1);
y = matches(2);

% Define the width and height of the rectangle
width = 47;
height = 17;

% Draw the rectangle
for i = 1:size(matches, 1)
    x = matches(i, 1);
    y = matches(i, 2);

    rectangle('Position', [x, y, width, height], 'EdgeColor', 'r', 'LineWidth', 2);
end

hold off; % Release the hold on the figure


%% 

function [model, falsePositives] = trainCascadeStage(datafeatures, dataclass, itt)
    % Train the AdaBoost model
    [~, model] = adaboost('train', datafeatures, dataclass, itt);

    % Apply the trained model to the data
    estimateclass = adaboost('apply', datafeatures, model);
    disp(size(estimateclass));
    estimateclass = estimateclass';

    % Identify false positives: Negative examples incorrectly classified as positive
    falsePositives = datafeatures(estimateclass == 1 & dataclass == -1, :);
end  

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

function estimateclass = applyAdaboostModelToImage(model, inputImage, haarFeatureMapping)
    % Preprocess the input image (this should match your training preprocessing)
    % For example, if you used integral images:
    % processedImage = integralImage(inputImage);

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
            midY = pixelY + floor(dimY/2);
            topSum = rectSum(pixelX, pixelY, pixelX + dimX, midY);
            bottomSum = rectSum(pixelX, midY, pixelX + dimX, pixelY + dimY);
            haarValue = topSum - bottomSum;

        case 2 % Vertical two-rectangle feature
            midX = pixelX + dimX/2;
            leftSum = rectSum(pixelX, pixelY, midX, pixelY + dimY);
            rightSum = rectSum(midX, pixelY, pixelX + dimX, pixelY + dimY);
            haarValue = leftSum - rightSum;

        case 3 % Horizontal three-rectangle feature
            thirdY = dimY / 3;
            topSum = rectSum(pixelX, pixelY, pixelX + dimX, pixelY + thirdY);
            middleSum = rectSum(pixelX, pixelY + thirdY, pixelX + dimX, pixelY + 2*thirdY);
            bottomSum = rectSum(pixelX, pixelY + 2*thirdY, pixelX + dimX, pixelY + dimY);
            haarValue = topSum - 2*middleSum + bottomSum;

        case 4 % Vertical three-rectangle feature
            thirdX = dimX / 3;
            leftSum = rectSum(pixelX, pixelY, pixelX + thirdX, pixelY + dimY);
            middleSum = rectSum(pixelX + thirdX, pixelY, pixelX + 2*thirdX, pixelY + dimY);
            rightSum = rectSum(pixelX + 2*thirdX, pixelY, pixelX + dimX, pixelY + dimY);
            haarValue = leftSum - 2*middleSum + rightSum;

        case 5 % Four-rectangle feature
            midX = pixelX + dimX/2;
            midY = pixelY + dimY/2;
            topLeftSum = rectSum(pixelX, pixelY, midX, midY);
            topRightSum = rectSum(midX, pixelY, pixelX + dimX, midY);
            bottomLeftSum = rectSum(pixelX, midY, midX, pixelY + dimY);
            bottomRightSum = rectSum(midX, midY, pixelX + dimX, pixelY + dimY);
            haarValue = topLeftSum - topRightSum - bottomLeftSum + bottomRightSum;
    end
end

function [estimateclasstotal,model]=adaboost(mode,datafeatures,dataclass_or_model,itt)
% This function AdaBoost, consist of two parts a simpel weak classifier and
% a boostitg part:
% The weak classifier tries to find the best treshold in one of the data
% dimensions to sepparate the data into two classes -1 and 1
% The boosting part calls the clasifier iteratively, after every classification
% step it changes the weights of miss-classified examples. This creates a
% cascade of "weak classifiers" which behaves like a "strong classifier"
%
%  Training mode:
%    [estimateclass,model]=adaboost('train',datafeatures,dataclass,itt)
%  Apply mode:
%    estimateclass=adaboost('apply',datafeatures,model)
% 
%  inputs/outputs:
%    datafeatures : An Array with size number_samples x number_features
%    dataclass : An array with the class off all examples, the class
%                 can be -1 or 1
%    itt : The number of training itterations
%    model : A struct with the cascade of weak-classifiers
%    estimateclass : The by the adaboost model classified data
%
%  Function is written by D.Kroon University of Twente (August 2010)
switch(mode)
    case 'train'
        % Train the adaboost model
        
        % Set the data class 
        dataclass=dataclass_or_model(:);
        model=struct;
        
        % Weight of training samples, first every sample is even important
        % (same weight)
        D=ones(length(dataclass),1)/length(dataclass);
        
        % This variable will contain the results of the single weak
        % classifiers weight by their alpha
        estimateclasssum=zeros(size(dataclass));
        
        % Calculate max min of the data
        boundary=[min(datafeatures,[],1) max(datafeatures,[],1)];
        % Do all model training itterations
        for t=1:itt
            % Find the best treshold to separate the data in two classes
            [estimateclass,err,h] = WeightedThresholdClassifier(datafeatures,dataclass,D);
            % Weak classifier influence on total result is based on the current
            % classification error
            alpha=1/2 * log((1-err)/max(err,eps));
            
            % Store the model parameters
            model(t).alpha = alpha;
            model(t).dimension=h.dimension;
            model(t).threshold=h.threshold;
            model(t).direction=h.direction;
            model(t).boundary = boundary;
            % We update D so that wrongly classified samples will have more weight
            D = D.* exp(-model(t).alpha.*dataclass.*estimateclass);
            D = D./sum(D);
            
            % Calculate the current error of the cascade of weak
            % classifiers
            estimateclasssum=estimateclasssum +estimateclass*model(t).alpha;
            estimateclasstotal=sign(estimateclasssum);
            model(t).error=sum(estimateclasstotal~=dataclass)/length(dataclass);
            if(model(t).error==0), break; end
        end
        
    case 'apply' 
        % Apply Model on the test data
        model=dataclass_or_model;
        
        % Limit datafeatures to orgininal boundaries
        if(length(model)>1);
            minb=model(1).boundary(1:end/2);
            maxb=model(1).boundary(end/2+1:end);
            datafeatures=bsxfun(@min,datafeatures,maxb);
            datafeatures=bsxfun(@max,datafeatures,minb);
        end
    
        % Add all results of the single weak classifiers weighted by their alpha 
        estimateclasssum=zeros(size(datafeatures,1),1);
        for t=1:length(model);
            estimateclasssum=estimateclasssum+model(t).alpha*ApplyClassTreshold(model(t), datafeatures);
        end
        % If the total sum of all weak classifiers
        % is less than zero it is probablly class -1 otherwise class 1;
        estimateclasstotal=sign(estimateclasssum);
        
    otherwise
        error('adaboost:inputs','unknown mode');
end
end
function [estimateclass,err,h] = WeightedThresholdClassifier(datafeatures,dataclass,dataweight)
% This is an example of an "Weak Classifier", it caculates the optimal
% threshold for all data feature dimensions.
% It then selects the dimension and  treshold which divides the 
% data into two class with the smallest error.
% Number of treshold steps
ntre=2e3;
% Split the data in two classes 1 and -1
r1=datafeatures(dataclass<0,:); w1=dataweight(dataclass<0);
r2=datafeatures(dataclass>0,:); w2=dataweight(dataclass>0);
% Calculate the min and max for every dimensions
minr=min(datafeatures,[],1)-1e-10; maxr=max(datafeatures,[],1)+1e-10;
% Make a weighted histogram of the two classes
p2c= ceil((bsxfun(@rdivide,bsxfun(@minus,r2,minr),(maxr-minr)))*(ntre-1)+1+1e-9);   p2c(p2c>ntre)=ntre;
p1f=floor((bsxfun(@rdivide,bsxfun(@minus,r1,minr),(maxr-minr)))*(ntre-1)+1-1e-9);  p1f(p1f<1)=1;
ndims=size(datafeatures,2);
i1=repmat(1:ndims,size(p1f,1),1);  i2=repmat(1:ndims,size(p2c,1),1);
h1f=accumarray([p1f(:) i1(:)],repmat(w1(:),ndims,1),[ntre ndims],[],0);
h2c=accumarray([p2c(:) i2(:)],repmat(w2(:),ndims,1),[ntre ndims],[],0);
% This function calculates the error for every all possible treshold value
% and dimension
h2ic=cumsum(h2c,1);
h1rf=cumsum(h1f(end:-1:1,:),1); h1rf=h1rf(end:-1:1,:);
e1a=h1rf+h2ic;
e2a=sum(dataweight)-e1a;
% We want the treshold value and dimension with the minimum error
[err1a,ind1a]=min(e1a,[],1);  dim1a=(1:ndims); dir1a=ones(1,ndims);
[err2a,ind2a]=min(e2a,[],1);  dim2a=(1:ndims); dir2a=-ones(1,ndims);
A=[err1a(:),dim1a(:),dir1a(:),ind1a(:);err2a(:),dim2a(:),dir2a(:),ind2a(:)];
[err,i]=min(A(:,1)); dim=A(i,2); dir=A(i,3); ind=A(i,4);
thresholds = linspace(minr(dim),maxr(dim),ntre);
thr=thresholds(ind);
% Apply the new treshold
h.dimension = dim; 
h.threshold = thr; 
h.direction = dir;
estimateclass=ApplyClassTreshold(h,datafeatures);
end
function y = ApplyClassTreshold(h, x)
% Draw a line in one dimension (like horizontal or vertical)
% and classify everything below the line to one of the 2 classes
% and everything above the line to the other class.
if(h.direction == 1)
    y =  double(x(:,h.dimension) >= h.threshold);
else
    y =  double(x(:,h.dimension) < h.threshold);
end
y(y==0) = -1;
end


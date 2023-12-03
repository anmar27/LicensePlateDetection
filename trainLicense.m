%Training Code

clc;
clear all;

%Creation of "saving" arrays in case of paused training
save1 =[];save2=[]; save3=[]; save4=[]; save5=[];

positiveImagesFolder = 'C:\Users\Usuario\OneDrive - Hanzehogeschool Groningen\Escritorio\Matlab Uni\CV-Project\Project CV\matlab-viola-jones\trainHaar\positiveImages1';
negativeImagesFolder = 'C:\Users\Usuario\OneDrive - Hanzehogeschool Groningen\Escritorio\Matlab Uni\CV-Project\Project CV\matlab-viola-jones\trainHaar\negativeimages1';

%positiveSize = numel(dir(positiveImagesFolder))-3;
%negativeSize = numel(dir(negativeImagesFolder))-2;
positiveSize = 100;
negativeSize = 200;

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
    for scaleWidth = 1:(imageWidth / baseHaarWidth)
        for scaleHeight = 1:(imageHeight / baseHaarHeight)
            haarWidth = baseHaarWidth * scaleWidth;
            haarHeight = baseHaarHeight * scaleHeight;
            
            % Number of positions where this scaled Haar feature can fit in the image
            positionsX = imageWidth - haarWidth + 1;
            positionsY = imageHeight - haarHeight + 1;

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

% Loop over each image
for imgIndex = 1:totalImages
    currentImage = allImages{imgIndex};
    integralImage = currentImage; % Already an integral image

    % Initialize a feature index for each image
    featureIndex = 1;

    % Iterate over Haar types
    for haarType = 1:size(haars, 1)
        dimX = haars(haarType, 1);
        dimY = haars(haarType, 2);

        % Iterate over positions and scales within the image
        for pixelX = 1:(imageWidth - dimX)
            for pixelY = 1:(imageHeight - dimY)
                % Calculate Haar feature value
                haarValue = calcHaarValues(integralImage, haarType, pixelX, pixelY, dimX, dimY);
                fprintf('Haar Type: %d, Value: %f, dimX:%d, dimY:%d, pixelX:%d, pixelY:%d\n', haarType, haarValue,dimX,dimY,pixelX,pixelY);
                % Store in datafeatures matrix
                datafeatures(imgIndex, featureIndex) = haarValue;

                % Update featureIndex
                featureIndex = featureIndex + 1;
            end
        end
    end
end

%Limiting to 3433 since we are not sonsidering resized features
datafeatures = datafeatures(1:300,1:3433);

%% 
% Number of iterations for AdaBoost
itt = 3;

% Train the AdaBoost model
[estimateclasstotal, model] = adaboost('train', datafeatures, dataclass, itt);
%% 

% Calculate the training accuracy
accuracy = sum(dataclass == estimateclasstotal) / length(dataclass);
fprintf('Training Accuracy: %.2f%%\n', accuracy * 100);
%% 
%Sliding window approach to find 
inputImagePath = "C:\Users\Usuario\OneDrive - Hanzehogeschool Groningen\Escritorio\Matlab Uni\CV-Project\Project CV\001\CroppedVehicles\03490.jpg_vehicle_6.jpg";
if isfile(inputImagePath)
    % Load the image
    inputImage = imread(inputImagePath);
end
matches = [];
windowSize = [17, 47];
stepSize = 5;
fprintf("Starting SW approach\n");

% Iterate over the image with the sliding window
for x = 1:stepSize:(size(inputImage, 2) - windowSize(2))
    for y = 1:stepSize:(size(inputImage, 1) - windowSize(1))
        % Extract the window
        fprintf("Iaminside:)\n");
        window = inputImage(y:y+windowSize(1)-1, x:x+windowSize(2)-1);

        % Extract features from this window
        windowFeatures = getHaarFeatures(window); % Define this function based on your feature extraction method

        % Classify the window using the AdaBoost model
        classificationScore = adaboost('apply', windowFeatures, model);
        fprintf("%d",classificationScore)
        % Check if the score is above a certain threshold
        if classificationScore == 1
            fprintf("Possible match in steps x:%d & y:%d",x,y);
            matches = [matches; x, y, classificationScore];
            % This window is a potential match
            % Store the window coordinates, size, and score for further processing
        end
    end
end


%% 
% Assuming new_datafeatures is your new data
new_estimateclass = adaboost('apply', datafeatures, model);

% Display the predicted classes
disp(new_estimateclass);


  %% 
  
 % Use Adaboost to make a classifier
  [classestimate,model]=adaboost('train',datafeatures,dataclass,2);
 % Training results
 % Show results
  negative=datafeatures(classestimate==-1,:); positive=datafeatures(classestimate==1,:);
  I=zeros(161,161);
  for i=1:length(model)
      if(model(i).dimension==1)
          if(model(i).direction==1), rec=[-80 -80 80+model(i).threshold 160];
          else rec=[model(i).threshold -80 80-model(i).threshold 160 ];
          end
      else
          if(model(i).direction==1), rec=[-80 -80 160 80+model(i).threshold];
          else rec=[-80 model(i).threshold 160 80-model(i).threshold];
          end
      end
      rec=round(rec);
      y=rec(1)+81:rec(1)+81+rec(3); x=rec(2)+81:rec(2)+81+rec(4);
      I=I-model(i).alpha; I(x,y)=I(x,y)+2*model(i).alpha;    
  end
 subplot(2,2,2), imshow(I,[]); colorbar; axis xy;
 colormap('jet'), hold on
 plot(negative(:,1)+81,negative(:,2)+81,'bo');
 plot(positive(:,1)+81,positive(:,2)+81,'ro');
 title('Training Data classified with adaboost model');
 % Show the error verus number of weak classifiers
 error=zeros(1,length(model)); for i=1:length(model), error(i)=model(i).error; end 
 subplot(2,2,3), plot(error); title('Classification error versus number of weak classifiers');
 % Make some test data
  angle=rand(200,1)*2*pi; l=rand(200,1)*70; testdata=[sin(angle).*l cos(angle).*l];
 % Classify the testdata with the trained model
  testclass=adaboost('apply',testdata,model);
 % Show result
  negative=testdata(testclass==-1,:); positive=testdata(testclass==1,:);
 % Show the data
  subplot(2,2,4), hold on
  plot(negative(:,1),negative(:,2),'b*');
  plot(positive(:,1),positive(:,2),'r*');
  axis equal;
  title('Test Data classified with adaboost model');

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

function val = calcHaarVal(img,haar,pixelX,pixelY,haarX,haarY)
% img: integral image of an input image
% haar: which Haar feature (1-5)
% pixelX/Y: start point in (X,Y)
% haarX/Y: Haar feature size in X and Y directions

% getCorners() finds the total of the pixel intensity values in a white/black "box"
    moveX = haarX-1;
    moveY = haarY-1;
   
    
    if haar == 1 % top/down white-black
        white = getCorners(img,pixelX,pixelY,pixelX+moveX,pixelY+floor(moveY/2));
        test = floor(moveY/2);
        fprintf("%f",test);
        black = getCorners(img,pixelX,pixelY+ceil(moveY/2),pixelX+moveX,pixelY+moveY);
        val = white-black;
    elseif haar == 2 % left/right white-black
        white = getCorners(img,pixelX,pixelY,pixelX+floor(moveX/2),pixelY+moveY);
        black = getCorners(img,pixelX+ceil(moveX/2),pixelY,pixelX+moveX,pixelY+moveY);
        val = white-black;
    elseif haar == 3 % top/mid/bottom white-black-white
        white1 = getCorners(img,pixelX,pixelY,pixelX+moveX,pixelY+floor(moveY/3));
        black = getCorners(img,pixelX,pixelY+ceil(moveY/3),pixelX+moveX,pixelY+floor((moveY)*(2/3)));
        white2 = getCorners(img,pixelX,pixelY+ceil((moveY)*(2/3)),pixelX+moveX,pixelY+moveY);
        val = white1 + white2 - black;
    elseif haar == 4 % left/mid/right white-black-white
        white1 = getCorners(img,pixelX,pixelY,pixelX+floor(moveX/3),pixelY+moveY);
        black = getCorners(img,pixelX+ceil(moveX/3),pixelY,pixelX+floor((moveX)*(2/3)),pixelY+moveY);
        white2 = getCorners(img,pixelX+ceil((moveX)*(2/3)),pixelY,pixelX+moveX,pixelY+moveY);
        val = white1 + white2 - black;
    elseif haar == 5 % checkerboard-style white-black-white-black
        white1 = getCorners(img,pixelX,pixelY,pixelX+floor(moveX/2),pixelY+floor(moveY/2));
        black1 = getCorners(img,pixelX+ceil(moveX/2),pixelY,pixelX+moveX,pixelY+floor(moveY/2));
        black2 = getCorners(img,pixelX,pixelY+ceil(moveY/2),pixelX+floor(moveX/2),pixelY+moveY);
        white2 = getCorners(img,pixelX+ceil(moveX/2),pixelY+ceil(moveY/2),pixelX+moveX,pixelY+moveY);
        val = white1+white2-(black1+black2);
    end
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
            midY = pixelY + dimY/2;
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
    % a boosting part:
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
    %  %% Example
    %
    %  example.m
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
    ntre=2e5;
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


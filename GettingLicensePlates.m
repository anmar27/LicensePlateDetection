%Code that creates images of licenses from vehicles dataset

clear height width

% Specify the directory where your images are stored
imageDir = 'D:\DATASET MATLAB\001';

% Specify the path to your CSV file
csvFile = 'D:\DATASET MATLAB\out_001_lp.csv';
% Specify the directory where you want to save cropped images
outputDir = 'D:\DATASET MATLAB\croppedPlates001';

% Create the output directory if it doesn't exist
if ~exist(outputDir, 'dir')
    mkdir(outputDir)
end

% Read the CSV file
data = readtable(csvFile);

% Loop over each row in the table
for i = 1:height(data)
    % Read the image
    imgName = data.image{i};
    imgPath = fullfile(imageDir, imgName);
    img = imread(imgPath);

    % Get the cropping coordinates
    xMin = data.xMin(i);
    yMin = data.yMin(i);
    xMax = data.xMax(i);
    yMax = data.yMax(i);

    w= xMax - xMin;
    h = yMax - yMin;

    % Crop the image
    if w > 27 && h > 17
        croppedImg = imcrop(img, [xMin, yMin, w, h]);

        % Save the cropped image in the output directory
        [~, name, ext] = fileparts(imgName);
        outputFileName = sprintf('%s_cropped_%d%s', name, data.id(i), ext);
        imwrite(croppedImg, fullfile(outputDir, outputFileName));
    end
end
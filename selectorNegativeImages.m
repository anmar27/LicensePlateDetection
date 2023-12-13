clc;
clear all;

positive_or_negative= "negatives";
n_crops = 4;

% Folder where cropped vehicles are stored
folderPath = ['..\CV-Project\Project CV\002']; 
subDirPath = fullfile(folderPath,positive_or_negative); % Subdirectory for negatives

% Get a list of all files in the folder
imageFiles = dir(fullfile(folderPath, '*.jpg'));
nnn

% Loop through each file
for i = 1:length(imageFiles)

    close all;
    img = imread(fullfile(folderPath, imageFiles(i).name));
    
    % Display the image
    hFig = figure;
    imshow(img);
    title(['Press any key to skip or wait to select regions for ', imageFiles(i).name]);
    
    % Wait for a button press to decide to skip or continue
    waitforbuttonpress;
    key = get(hFig, 'CurrentCharacter');
    
    % Check if the pressed key is 'n' (for next)
    if lower(key) == 'n'
        close(hFig); % Close the figure window
        continue; % Skip to the next image
    end
    
    
    % Display the image and let user select three regions
    for j = 1:n_crops
        figure;
        imshow(img);
        title(['Select region ', num2str(j), ' for ', imageFiles(i).name]);
        
        % Let user draw a rectangle on the image
        rect = getrect;
   
        
        % Crop the image
        croppedImg = imcrop(img, rect);
        
        % Resize the cropped image to exactly 17x47 pixels, if necessary
        croppedImg = imresize(croppedImg, [17, 47]);        
        
        %Turn into gray scale
        croppedImg = rgb2gray(croppedImg);              

        % Save the cropped image
        croppedFilename = fullfile(subDirPath, [positive_or_negative,'_', num2str(i), '_', num2str(j), '.jpg']);
        imwrite(croppedImg, croppedFilename);
    end
    
end

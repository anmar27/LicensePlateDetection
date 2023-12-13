%Code that generates cropped vehicles for later use for negative images

% Define the path to the directory containing images
folderPath = '../Project CV\002';

% List all JPG files in the folder
imageFiles = dir(fullfile(folderPath, '*.jpg'));

% Initialize the detector (assuming you have a pre-trained detector)
 detector = vehicleDetectorYOLOv2();

% Process each image file
for i = 1:length(imageFiles)
    % Read the image
    imagePath = fullfile(folderPath, imageFiles(i).name);
    I = imread(imagePath);

    % Detect vehicles
    [bboxes, scores] = detect(detector, I);
    I = insertObjectAnnotation(I, 'rectangle', bboxes, scores);

    % Extract areas of interest
    n_vehicles = size(bboxes, 1);

    for n = 1:n_vehicles
        if bboxes(n,3) >= 150 && bboxes(n,4) >= 150
            vehicle = imcrop(I, [bboxes(n,1), bboxes(n,2), bboxes(n,3), bboxes(n,4)]);
            % Construct a filename for the cropped image
            filename = sprintf('%s_vehicle_%d.jpg', imageFiles(i).name, n);
        
            % Define full path to save the cropped image
            savePath = fullfile(folderPath, 'CroppedVehicles', filename);

            % Save the cropped image
            imwrite(vehicle, savePath);
        end
    end
end

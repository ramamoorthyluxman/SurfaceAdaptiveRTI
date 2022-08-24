function [Image] = readImg(path, Name, index)

    path_of_file = strcat(path, Name{index});
    %if contains(path_of_file, '.png')
     %   buffer = tonemap(hdrread(path_of_file));
    if contains(path_of_file, '.jpg')
        buffer = imread(path_of_file);
    end
    
    Image = single(buffer(:,:,1));
    
end
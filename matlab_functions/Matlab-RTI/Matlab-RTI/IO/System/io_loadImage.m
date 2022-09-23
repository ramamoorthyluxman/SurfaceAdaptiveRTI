function [Image] = io_loadImage(Path, Name, Type, Channel)

    if nargin < 3
        Type = 'single';
    end
    
    if nargin < 4
        Channel = 'Gray';
    end
    
    Path_of_file = fullfile(Path, Name);
    
    Imread = [".png", ".tiff", ".jpg"];
    HDRread = [".hdr"];
    if contains(Path_of_file, HDRread)
        buffer = hdrread(Path_of_file);
        buffer = buffer(:,:,1);
        [m, n] = size(buffer);
        buffer = reshape(buffer(:), n, m)';
    elseif contains(Path_of_file, Imread)
        buffer = imread(Path_of_file);
        if size(buffer, 3) == 3
            if strcmp(Channel, 'Red')
                buffer = buffer(:,:,1);
            elseif strcmp(Channel, 'Green')
                buffer = buffer(:,:,2);
            elseif strcmp(Channel, 'Blue')
                buffer = buffer(:,:,3);
            elseif strcmp(Channel, 'Gray')
                buffer = mean(buffer, 3);
            end
        end
    end
    
    if strcmp(Type, 'single')
        Image = single(buffer);
    elseif strcmp(Type, 'uint8')
        Image = uint8(buffer);
    end
end
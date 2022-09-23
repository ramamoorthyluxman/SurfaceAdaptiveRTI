function [Acquisition] = process_cropAcquisition(Acquisition, rectangle)

    Acquisition.Data = reshape(Acquisition.Data, size(Acquisition.Data, 1), Acquisition.Data_Size(1), Acquisition.Data_Size(2));
    Acquisition.Data = Acquisition.Data(:, rectangle(1):rectangle(1)+rectangle(3)-1, rectangle(2):rectangle(2)+rectangle(4)-1);
    Acquisition.Data = reshape(Acquisition.Data, size(Acquisition.Data, 1), []) ;
    Acquisition.Data_Size(1) = rectangle(3);
    Acquisition.Data_Size(2) = rectangle(4);
    Acquisition.Path = fullfile(Acquisition.Path, 'Cropped', strcat(num2str(rectangle(1)), '_', num2str(rectangle(2))));
    
end
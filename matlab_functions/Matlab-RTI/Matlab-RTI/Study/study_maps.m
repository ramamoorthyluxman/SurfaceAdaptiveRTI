function [Maps1, Maps2] = study_maps(Acquisition1, Acquisition2)
    
    Acquisition1.Maps = map_generateAll(Acquisition1.Data, Acquisition1.Data_Size, Acquisition1.Pixel_Size, Acquisition1.LP);
    
    io_saveMaps(Acquisition1, 'parula');
    io_saveMaps(Acquisition1, 'gray');

    Maps1 = Acquisition1.Maps;
    
    if nargin == 2
        Acquisition2.Maps = map_generateAll(Acquisition2.Data, Acquisition2.Data_Size, Acquisition2.Pixel_Size, Acquisition2.LP);
        
        io_saveMaps(Acquisition1, Acquisition2, 'parula');
        io_saveMaps(Acquisition1, Acquisition2, 'gray');
    else
        Maps2 = [];
    end

    
    
end


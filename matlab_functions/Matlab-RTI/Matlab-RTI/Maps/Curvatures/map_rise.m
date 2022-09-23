function [Maps] = map_rise(Maps, Data, LP, Pixel_Size)
    
    Maps = map_riseX(Maps, Data, LP, Pixel_Size);
    Maps = map_riseY(Maps, Data, LP, Pixel_Size);
    Maps = map_riseXY(Maps, Data, LP, Pixel_Size);

end


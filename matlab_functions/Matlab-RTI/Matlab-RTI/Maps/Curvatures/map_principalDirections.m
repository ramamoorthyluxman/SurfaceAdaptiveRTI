function [Maps] = map_principalDirections(Maps, Data, LP, Data_Size)
    
    Maps = map_KminVector(Maps, Data, LP, Data_Size);
    Maps = map_KmaxVector(Maps, Data, LP, Data_Size);

end


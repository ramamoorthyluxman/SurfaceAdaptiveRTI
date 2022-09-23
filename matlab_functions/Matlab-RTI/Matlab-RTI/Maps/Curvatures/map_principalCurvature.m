function [Maps] = map_principalCurvature(Maps, Data, LP, Data_Size)
    
    Maps = map_kmin(Maps, Data, LP, Data_Size);
    Maps = map_kmax(Maps, Data, LP, Data_Size);

end


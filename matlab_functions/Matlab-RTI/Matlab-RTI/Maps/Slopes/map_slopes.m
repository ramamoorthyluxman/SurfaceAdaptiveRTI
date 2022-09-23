function [Maps] = map_slopes(Maps, Data, LP)

    Maps = map_dx(Maps, Data, LP);
    Maps = map_dy(Maps, Data, LP);
    Maps = map_dz(Maps, Data, LP);

end
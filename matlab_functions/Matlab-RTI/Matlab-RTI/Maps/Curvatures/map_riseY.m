function [Maps] = map_riseY(Maps, Data, LP, Pixel_Size)
    
    if ~map_isComputed(Maps, 'RiseY')
    
        Maps = map_normal(Maps, Data, LP);
        Maps.Data.('RiseY') = -(Pixel_Size(2) .* Maps.Data.('Normal')(2, :)) ./ Maps.Data.('Normal')(3,:);
    
    end

end


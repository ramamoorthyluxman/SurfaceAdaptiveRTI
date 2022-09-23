function [Maps] = map_riseXY(Maps, Data, LP, Pixel_Size)
    
    if ~map_isComputed(Maps, 'RiseXY')
    
        Maps = map_normal(Maps, Data, LP);
        Maps.Data.('RiseXY') = -(Pixel_Size(1) .* Maps.Data.('Normal')(1, :) + Pixel_Size(2) .* Maps.Data.('Normal')(2, :)) ./ Maps.Data.('Normal')(3,:);
    
    end
end


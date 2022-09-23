function [Maps] = map_riseX(Maps, Data, LP, Pixel_Size)
    
    if ~map_isComputed(Maps, 'RiseX')
    
        Maps = map_normal(Maps, Data, LP);
        Maps.Data.('RiseX') = -(Pixel_Size(1) .* Maps.Data.('Normal')(1, :)) ./ Maps.Data.('Normal')(3,:);
    
    end
end


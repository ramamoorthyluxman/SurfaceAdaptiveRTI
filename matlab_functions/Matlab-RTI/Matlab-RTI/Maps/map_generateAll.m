function [Maps] = map_generateAll(Maps, Data, LP, Data_Size, Pixel_Size)

    maps_names = fieldnames(Maps);

    for i=1:size(maps_names)
        Maps = map_generateByName(maps_names{i}, Maps, Data, 'LP', LP, 'DataSize', Data_Size, 'PixelSize', Pixel_Size);
    end
    
end


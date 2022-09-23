function [Maps] = map_dz(Maps, Data, LP)
    
    if ~map_isComputed(Maps, 'Dz')

        Maps = map_normal(Maps, Data, LP);
        Maps.Data.('Dz') = Maps.Data.('Normal')(3, :);

    end
    
end


function [Maps] = map_dx(Maps, Data, LP)
    
    if ~map_isComputed(Maps, 'Dx')

        Maps = map_normal(Maps, Data, LP);
        Maps.Data.('Dx') = Maps.Data.('Normal')(1, :);

    end

end


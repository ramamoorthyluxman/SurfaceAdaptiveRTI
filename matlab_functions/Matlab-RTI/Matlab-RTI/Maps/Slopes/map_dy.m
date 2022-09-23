function [Maps] = map_dy(Maps, Data, LP)

    if ~map_isComputed(Maps, 'Dy')

        Maps = map_normal(Maps, Data, LP);
        Maps.Data.('Dy') = Maps.Data.('Normal')(2, :);

    end
    
end


function [Maps] = map_dipAngle(Maps, Data, LP)
    
    if ~map_isComputed(Maps, 'DipAngle')

        Maps = map_normal(Maps, Data, LP);
        [~, Phi] = process_carthesian2polar(Maps.Data.('Normal')(1, :), Maps.Data.('Normal')(2, :), Maps.Data.('Normal')(3, :));
        Maps.Data.('DipAngle') = 90 - Phi;

    end
    
end


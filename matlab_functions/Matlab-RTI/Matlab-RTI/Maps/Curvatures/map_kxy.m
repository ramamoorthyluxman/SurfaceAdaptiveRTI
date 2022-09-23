function [Maps] = map_kxy(Maps, Data, LP, Data_Size)

    if ~map_isComputed(Maps, 'Kxy')

        Maps = map_dy(Maps, Data, LP);
        [Kxy,~] = gradient(reshape(Maps.Data.('Dy'), Data_Size(1), Data_Size(2)));
        Maps.Data.('Kxy') = reshape(Kxy, 1, Data_Size(1) * Data_Size(2));

    end
end


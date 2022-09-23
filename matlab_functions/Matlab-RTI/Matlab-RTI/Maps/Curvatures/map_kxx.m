function [Maps] = map_kxx(Maps, Data, LP, Data_Size)
    
    if ~map_isComputed(Maps, 'Kxx')

        Maps = map_dx(Maps, Data, LP);
        [Kxx,~] = gradient(reshape(Maps.Data.('Dx'), Data_Size(1), Data_Size(2)));
        Maps.Data.('Kxx') = reshape(Kxx, 1, Data_Size(1) * Data_Size(2));

    end

end


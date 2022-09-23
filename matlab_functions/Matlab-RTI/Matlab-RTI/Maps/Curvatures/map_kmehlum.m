function [Maps] = map_kmehlum(Maps, Data, LP, Data_Size)
    
    if ~map_isComputed(Maps, 'KMehlum')
    
        Maps = map_kmean(Maps, Data, LP, Data_Size);
        Maps = map_kgaussian(Maps, Data, LP, Data_Size);

        Maps.Data.('KMehlum') = 3.0 * Maps.Data.('Kmean') .* Maps.Data.('Kmean') / 2.0 - Maps.Data.('Kgaussian') / 2.0;

    end

end


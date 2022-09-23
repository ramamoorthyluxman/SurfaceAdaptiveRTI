function [Maps] = map_kgaussian(Maps, Data, LP, Data_Size)
    
    if ~map_isComputed(Maps, 'Kgaussian')

        Maps = map_principalCurvature(Maps, Data, LP, Data_Size);
        Maps.Data.('Kgaussian') = Maps.Data.('Kmin') .* Maps.Data.('Kmax');

    end

end
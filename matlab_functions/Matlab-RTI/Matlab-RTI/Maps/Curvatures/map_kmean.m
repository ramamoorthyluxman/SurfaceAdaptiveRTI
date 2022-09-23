function [Maps] = map_kmean(Maps, Data, LP, Data_Size)
    
    if ~map_isComputed(Maps, 'Kmean')

        Maps = map_principalCurvature(Maps, Data, LP, Data_Size);
        Maps.Data.('Kmean') = ( Maps.Data.('Kmin') +  Maps.Data.('Kmax')) / 2.0;

    end

end
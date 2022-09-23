function [Maps] = map_curvedness(Maps, Data, LP, Data_Size)

        if ~map_isComputed(Maps, 'Curvedness')

            Maps = map_principalCurvature(Maps, Data, LP, Data_Size);
            Maps.Data.('Curvedness') = sqrt((Maps.Data.('Kmax') .* Maps.Data.('Kmax') + Maps.Data.('Kmin') .* Maps.Data.('Kmin')) / 2.0);

        end
        
end


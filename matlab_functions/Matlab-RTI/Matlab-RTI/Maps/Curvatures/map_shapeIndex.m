function [Maps] = map_shapeIndex(Maps, Data, LP, Data_Size)

    if ~map_isComputed(Maps, 'ShapeIndex')

        Maps = map_principalCurvature(Maps, Data, LP, Data_Size);
        Maps.Data.('ShapeIndex') = -2 * atan((Maps.Data.('Kmax') + Maps.Data.('Kmin')) ./ (Maps.Data.('Kmax') - Maps.Data.('Kmin'))) / pi;

    end
end


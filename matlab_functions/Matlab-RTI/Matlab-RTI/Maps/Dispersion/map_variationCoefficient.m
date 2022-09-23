function [Maps] = map_variationCoefficient(Maps, Data)

    if ~map_isComputed(Maps, 'VariationCoefficient')

        Maps = map_standardDeviation(Maps, Data);
        Maps = map_mean(Maps, Data);

        Maps.Data.('VariationCoefficient') = Maps.Data.('StandardDeviation') ./ Maps.Data.('Mean');

    end

end
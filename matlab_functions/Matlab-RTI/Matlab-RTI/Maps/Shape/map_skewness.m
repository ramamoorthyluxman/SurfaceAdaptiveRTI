function [Maps] = map_skewness(Maps, Data)

    if ~map_isComputed(Maps, 'Skewness')

        Maps.Data.('Skewness') = skewness(Data, 1, 1);

    end

end
function [Maps] = map_kurtosis(Maps, Data)

    if ~map_isComputed(Maps, 'Kurtosis')

        Maps.Data.('Kurtosis') = kurtosis(Data, 0, 1);

    end

end
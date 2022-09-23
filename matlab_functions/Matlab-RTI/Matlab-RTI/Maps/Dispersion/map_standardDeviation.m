function [Maps] = map_standardDeviation(Maps, Data)
    
    if ~map_isComputed(Maps, 'StandardDeviation')

        Maps.Data.('StandardDeviation') = std(Data, 0, 1);

    end
end
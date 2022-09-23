function [Maps] = map_median(Maps, Data)

    if ~map_isComputed(Maps, 'Median')

        Maps.Data.('Median') = median(Data, 1);

    end

end
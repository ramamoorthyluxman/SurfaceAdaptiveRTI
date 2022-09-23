function [Maps] = map_max(Maps, Data)

    if ~map_isComputed(Maps, 'Max')

        Maps.Data.('Max') = max(Data, [], 1);

    end

end


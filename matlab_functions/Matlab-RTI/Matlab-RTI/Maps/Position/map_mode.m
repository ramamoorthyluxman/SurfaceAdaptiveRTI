function [Maps] = map_mode(Maps, Data)

    if ~map_isComputed(Maps, 'Mode')

        Maps.Data.('Mode') = mode(Data, 1);

    end

end


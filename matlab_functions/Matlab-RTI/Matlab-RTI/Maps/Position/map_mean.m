function [Maps] = map_mean(Maps, Data)

    if ~map_isComputed(Maps, 'Mean')

        Maps.Data.('Mean') = mean(Data, 1);

    end

end


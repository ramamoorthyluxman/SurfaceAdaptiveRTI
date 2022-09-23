function [Maps] = map_min(Maps, Data)

    if ~map_isComputed(Maps, 'Min')

        Maps.Data.('Min') = min(Data, [], 1);

    end

end

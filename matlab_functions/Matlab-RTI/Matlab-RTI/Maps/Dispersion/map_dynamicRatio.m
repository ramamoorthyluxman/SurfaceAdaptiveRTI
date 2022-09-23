function [Maps] = map_dynamicRatio(Maps, Data)
    
    if ~map_isComputed(Maps, 'DynamicRatio')
    
        Maps = map_max(Maps, Data);
        Maps = map_min(Maps, Data);

        Maps.Data.('DynamicRatio') = Maps.Data.('Max') / Maps.Data.('Min');
        Maps.Data.('DynamicRatio')(isinf(Maps.Data.('DynamicRatio'))) = max(Maps.Data.('DynamicRatio')(:));
    
    end

end


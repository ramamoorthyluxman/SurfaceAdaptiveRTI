function [IsComputed] = map_isComputed(Maps, MapName)
    
    IsComputed = ~isempty(Maps.Data.(MapName));

end


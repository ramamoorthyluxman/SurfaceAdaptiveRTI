function [Maps] = map_generateByNames(MapNames, Maps, Data, varargin)
   
    for i=1:length(MapNames)
        Maps = map_generateByName(MapNames{i}, Maps, Data, varargin{:});
    end
    
end
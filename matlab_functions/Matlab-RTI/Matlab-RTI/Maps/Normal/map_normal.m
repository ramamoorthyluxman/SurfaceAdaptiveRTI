function [Maps] = map_normal(Maps, Data, LP)
    
    if ~map_isComputed(Maps, 'Normal')

        Positions = [LP.X LP.Y LP.Z];
        Positions_inv = (Positions' * Positions)\Positions';
        Normal = Positions_inv * Data;

        Normal = process_normalizeLostDynamic(Normal);

        Maps.Data.('Normal') = Normal;

    end

end
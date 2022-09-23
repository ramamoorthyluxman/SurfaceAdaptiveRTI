function [NormalTrimmed] = map_normalTrimmed(Data, LP, Limits)
   
    NormalTrimmed = zeros(1, size(Data, 2));
    Positions = [LP.X LP.Y LP.Z];
    Positions_inv = (Positions' * Positions)\Positions';
    
    minLim = Limits(1);
    maxLim = Limits(2);
    
    parfor i=1:size(Data, 2)
        indices = find(Data(:,i) >= minLim && Data(:,i) <= maxLim);
        
        if isempty(indices)
            NormalTrimmed(i) = Positions_inv * Data(:, i);
        else
            NormalTrimmed(i) = Positions_inv * Data(indices, i);
        end
    end
    
    NormalTrimmed = process_normalizeLostDynamic(NormalTrimmed);

end
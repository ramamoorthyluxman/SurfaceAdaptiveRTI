function [Maps] = map_ptmCoefficients(Maps, Data, LP)

    
    %LP = process_normalizeLostDynamic([LP.X LP.Y LP.Z]')';
    %LP = process_normalizeLostDynamic([LP.X LP.Y LP.Z]);
    
    %l = cat(2, power(LP(:,1), 2), power(LP(:,2), 2), LP(:,1) .* LP(:,2), LP(:,1), LP(:,2), ones(size(LP(:,1))));
    l = cat(2, power(LP.X, 2), power(LP.Y, 2), LP.X .* LP.Y, LP.X, LP.Y, ones(size(LP.X)));
    
    Maps.('PTM').Coefficients = l\Data;
    
end


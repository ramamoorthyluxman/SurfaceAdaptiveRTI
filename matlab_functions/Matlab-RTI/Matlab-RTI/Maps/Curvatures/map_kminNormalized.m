function [KminNormalized] = map_kminNormalized(Data, Data_Size, LP)
    
    KminVector = map_kminVector(Data, Data_Size, LP);
    
    KminNormalized = cat(1, KminVector, zeros(1, size(Data, 2)));
    KminNormalized = process_normalizeLostDynamic(KminNormalized);

end


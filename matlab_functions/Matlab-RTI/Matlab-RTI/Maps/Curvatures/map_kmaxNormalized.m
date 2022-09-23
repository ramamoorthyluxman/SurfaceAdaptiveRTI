function [KmaxNormalized] = map_kmaxNormalized(Data, Data_Size, LP)
    
    KmaxVector = map_kmaxVector(Data, Data_Size, LP);
    
    KmaxNormalized = cat(1, KmaxVector, zeros(1, size(Data, 2)));
    KmaxNormalized = process_normalizeLostDynamic(KmaxNormalized);

end
function [normalized] = process_normalizeLostDynamic(Data)

    norm = sqrt(sum(power(Data, 2), 1));
    norm = repmat(norm, size(Data, 1), 1);
    
    normalized = Data ./ norm;
    
end


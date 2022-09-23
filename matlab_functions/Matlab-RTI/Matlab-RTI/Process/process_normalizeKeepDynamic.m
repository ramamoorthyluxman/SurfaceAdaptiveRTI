function [normalized] = process_normalizeKeepDynamic(Image)

    normalized = Image - min(Image(:));
    normalized = normalized ./ max(normalized(:));
    
end
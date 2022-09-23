function [Mirrored, DataSizeNew] = process_mirrorEdge(Data, DataSize, MarginSize)

    Data = reshape(Data, DataSize);
    Mirrored = padarray(Data, [MarginSize MarginSize], 'replicate', 'both');
    DataSizeNew = size(Mirrored);
    Mirrored = reshape(Mirrored, 1, DataSize(1) * DataSize(2));
    
end


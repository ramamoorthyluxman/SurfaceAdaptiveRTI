function [Data] = process_cutoff(Data, order)

    Mean = nanmean(Data);
    STD = nanstd(Data);
    
    cutoff = [Mean - STD * order, Mean + STD * order]; 
    Data(Data < cutoff(1)) = cutoff(1);
    Data(Data > cutoff(2)) = cutoff(2);

end
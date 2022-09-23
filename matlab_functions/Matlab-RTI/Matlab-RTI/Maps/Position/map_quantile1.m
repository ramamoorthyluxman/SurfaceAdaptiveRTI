function [Maps] = map_quantile1(Maps, Data)

    if ~map_isComputed(Maps, 'Quantile1')

        Maps.Data.('Quantile1') = quantile(Data, 0.25, 1);

    end

end

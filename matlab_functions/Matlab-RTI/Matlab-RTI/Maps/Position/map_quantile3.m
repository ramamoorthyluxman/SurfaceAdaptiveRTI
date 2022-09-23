function [Maps] = map_quantile3(Maps, Data)

    if ~map_isComputed(Maps, 'Quantile3')

        Maps.Data.('Quantile3') = quantile(Data, 0.75, 1);

    end

end

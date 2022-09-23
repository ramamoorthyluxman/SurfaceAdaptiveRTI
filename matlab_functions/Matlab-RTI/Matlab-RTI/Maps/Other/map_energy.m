function [Maps] = map_energy(Maps, Data)

    if ~map_isComputed(Maps, 'Energy')

        Maps.Data.('Energy') = sum(power(Data, 2),1);

    end

end
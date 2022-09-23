function [Maps] = map_kyy(Maps, Data, LP, Data_Size)

    if ~map_isComputed(Maps, 'Kyy')

        Maps = map_dy(Maps, Data, LP);
        [~,Kyy] = gradient(reshape(Maps.Data.('Dy'), Data_Size(1), Data_Size(2)));
        Maps.Data.('Kyy') = reshape(Kyy, 1, Data_Size(1) * Data_Size(2));

    end
end
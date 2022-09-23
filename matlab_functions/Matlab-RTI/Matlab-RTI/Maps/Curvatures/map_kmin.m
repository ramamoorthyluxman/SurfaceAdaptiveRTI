function [Maps] = map_kmin(Maps, Data, Data_Size, LP)
    
    if ~map_isComputed(Maps, 'Kmin')

        Maps = map_kxx(Maps, Data, LP, Data_Size);
        Maps = map_kxy(Maps, Data, LP, Data_Size);
        Maps = map_kyy(Maps, Data, LP, Data_Size);

        Tensor = [reshape(Maps.Data.('Kxx'), 1, 1, []) reshape(Maps.Data.('Kxy'), 1, 1, []); ...
                  reshape(Maps.Data.('Kxy'), 1, 1, []) reshape(Maps.Data.('Kyy'), 1, 1, [])];
        Kmin = zeros(1, size(Data, 2));
        parfor i=1:size(Data, 2)
            Eigen = eig(Tensor(:,:,i));
            Kmin(i) = min(Eigen);
        end
        Maps.Data.('Kmin') = Kmin;

    end
end


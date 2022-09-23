function [Maps] = map_kmax(Maps, Data, LP, Data_Size)
    
    if ~map_isComputed(Maps, 'Kmax')

        Maps = map_kxx(Maps, Data, LP, Data_Size);
        Maps = map_kxy(Maps, Data, LP, Data_Size);
        Maps = map_kyy(Maps, Data, LP, Data_Size);

        Tensor = [reshape(Maps.Data.('Kxx'), 1, 1, []) reshape(Maps.Data.('Kxy'), 1, 1, []); ...
                  reshape(Maps.Data.('Kxy'), 1, 1, []) reshape(Maps.Data.('Kyy'), 1, 1, [])];
        Kmax = zeros(1, size(Data, 2));
        parfor i=1:size(Data, 2)
            Eigen = eig(Tensor(:,:,i));
            Kmax(i) = max(Eigen);
        end
        Maps.Data.('Kmax') = Kmax;

    end
end


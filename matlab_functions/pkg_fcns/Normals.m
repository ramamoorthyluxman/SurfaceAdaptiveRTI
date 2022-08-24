
function Nrgb= Normals(height, width, val, nfiles, pixel_normalise,LP )

normal2 = zeros(height,width,3);

            W=ones(nfiles,1);
         
            %G= (LP'*diag(W)*LP)\ (LP' *diag(W) * pixel_normalise);
            G=pinv(LP)*pixel_normalise;
            normG = sqrt(G(1,:).^2 + G(2,:).^2 + G(3,:).^2);
            G(1,:)=G(1,:)./normG;
            G(2,:)=G(2,:)./normG;
            G(3,:)=G(3,:)./normG;
            normal2(:,:,1) = reshape(G(1,:), height,width);
            normal2(:,:,2) = reshape(G(2,:), height,width);
            normal2(:,:,3) = reshape(G(3,:), height,width);
            
            N = round(normal2*100)/100;
            Nrgb = zeros(height,width,3);
            % Color codes&
            Nrgb(:, :, 1)=(N(:,:, 1) + 1) / 2;
            Nrgb(:, :, 2)=(N(:,:, 2) + 1) / 2;
            Nrgb(:, :, 3)= N(:,:, 3);
            im_norm= Nrgb;  
   
end 
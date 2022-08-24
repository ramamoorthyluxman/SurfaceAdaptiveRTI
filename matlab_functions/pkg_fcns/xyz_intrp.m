function interpol= xyz_intrp(lp1,lp2,lp3,lp4, nfiles, vec_pos, pos)
    %for i=1:nfiles
   % xyz=[lp1; lp2; lp3; lp4];
    for i=1:nfiles
    xyz=[Images.LP1(i,:); Images.LP2(i,:); Images.LP3(i,:); Images.LP4(i,:)];
   % intr_1(i,:)=interp1((vec_pos),xyz,(pos(i)),'linear');  
    intr_2(i,1)= LeanInterp(vec_pos,xyz(:,1),pos(i)); 
    intr_2(i,2)= LeanInterp(vec_pos,xyz(:,2),pos(i)); 
    intr_2(i,3)= LeanInterp(vec_pos,xyz(:,3),pos(i)); 
   % interpol(:,:)=interp1(double(vec_pos),xyz,double(pos),'linear');    
    end
end
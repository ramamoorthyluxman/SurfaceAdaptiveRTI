function intr_xyz=inter_xyz(nfiles, LP1, LP2, LP3, LP4 , vec_pos, pos)
   % pos=pos;
    %vec_pos=vec_pos;
    %xyz=xyz_coor;
    for i=1:nfiles   
   % intr_1(i,:)=interp1((vec_pos),xyz,(pos(i)),'linear'); 
    xyz=[LP1(i,:); LP2(i,:); LP3(i,:); LP4(i,:)];
    intr_xyz(i,1)= LeanInterp(vec_pos,xyz(:,1),pos(i)); 
    intr_xyz(i,2)= LeanInterp(vec_pos,xyz(:,2),pos(i)); 
    intr_xyz(i,3)= LeanInterp(vec_pos,xyz(:,3),pos(i)); 
    %intr_2(i,:)= lininterp1(vec_pos,xyz,pos(i)); 
  
    end
end

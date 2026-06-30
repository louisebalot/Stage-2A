!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Function which calculates the vertical diffusion coefficient
! at a certain deep.
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
module m_diffcoeff
contains
function diffcoeff(deep,mixlay,totdepth)
   implicit none
   real diffcoeff
   real, intent(in)    :: deep     ! current depth
   real, intent(in)    :: mixlay   ! mixedlayer depth
   real, intent(in)    :: totdepth ! total depth

   real, parameter :: kkk=0.1                 ! thermocline sharpness
   real, parameter :: difftop=0.01*86400.0    ! Diffusion at surface (m^2/day)  
   real, parameter :: diffbot=0.0001*86400.0  ! Diffusion at bottom  (m^2/day)  
   
   diffcoeff=diffbot + (difftop-diffbot)*(atan(kkk*(mixlay-deep))&
             -atan(kkk*(mixlay-totdepth)))/(atan(kkk*(mixlay)) &
             -atan(kkk*(mixlay-totdepth)))

end function diffcoeff
end module

module m_getmxlay
contains
function getmxlay(MM,dim,time,dtref)
! Always pick the mxlayer from the first year stored in MM
   implicit none
   integer, intent(in) :: dim
   real, intent(in) :: MM(dim)
   real, intent(in) :: time
   real, intent(in) :: dtref
   real getmxlay
   real a,t
   integer n

   t=mod(time,365.0)
   n=int(t/dtref)+1
   a=(MM(n+1)-MM(n))/dtref
   
   getmxlay=MM(n)+a*(t-float(n-1)*dtref)
end function getmxlay
end module

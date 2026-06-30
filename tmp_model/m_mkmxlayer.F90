module m_mkmxlayer
contains
subroutine mkmxlayer(MM,dt)
! Generates the mixedlayer depth as a function of time
! Computation on over one year on a fixed grid which is
! independent of ndim,  results in identical smoothing
! and the same form of mxlayer when ndim and/or tfin is changed

   use mod_dimensions
   use m_shapiro
   use m_getmxlay
   implicit none

   real, intent(out) :: MM(ndim)
   real, intent(in) :: dt

   real, allocatable :: YMM(:)
   integer n
   real time

   integer, parameter :: ydim=400.0
   real,    parameter :: year=365.0
   real ydt
   allocate(YMM(ydim))

! First computes mixedlayer over one year on a fixed grid resolution

   ydt=year/float(ydim-1)
   do n=1,ydim              
      time=float(n-1)*ydt
      YMM(n)=mixldpth(time)
   enddo

! The smooth analytical mxlayer on fixed grid
   do n=1,200
      call shapiro(1,YMM,ydim)
   enddo

! Then interpolate mxlayer to MM(ndim)
   do n=1,ndim
      time=float(n-1)*dt
      MM(n)=getmxlay(YMM,ydim,time,ydt)
      write(44,*)time,MM(n)
   enddo

   deallocate(YMM)

contains
function mixldpth(time)
! function for computing the mixed layer depth
   use mod_dimensions
   implicit none
   real, intent(in) :: time
   real mixldpth
   real t


   real, parameter :: M0=62.0
   real, parameter :: M1=M0 + 0.30*60.0
   real, parameter :: M2=M1 - 3.05*(100.0-82.0)

   t=mod(time,365.0)

   select case (int(t))
   case (:59)
      mixldpth = M0 + 0.30*t
   case (60:81)
      mixldpth= M1
   case (82:99)
      mixldpth = M1 - 3.05*(t-82.0)
   case (100:239)
      mixldpth = M2
   case (240:364)
      mixldpth = M2 + 0.2952*(t-240.0)
   case default
       print *,'mixldpth: problem'
   stop
   end select

end function mixldpth
end subroutine mkmxlayer
end module

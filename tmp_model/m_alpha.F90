module m_alpha
contains
function alpha(deep,time,ppave)
! Function which calculates the growth rate = photosynthetic light,
! alpha for the different depths and phyto plankton shadows.

   use mod_alpha_pars
   implicit none
   real alpha

   real, intent(in) :: deep
   real, intent(in) :: time
   real, intent(in) :: ppave

   real aa
   real tau
   real delta
   real k1
   real JJ
   real A,B

   delta=(-0.406)*cos(2.0*pi*time/365.0)

   tau=acos(-tan(delta)*tan(psi))/2.0

!#if defined (BIAS)
!   aa=sin(delta)*sin(psi)*tau + cos(delta)*cos(psi)*sin(tau)
!   aa=aa*(3.0/8.0)*(1.-0.7*0.9)*(RR/pi)
!#else
   aa=sin(delta)*sin(psi)*tau + cos(delta)*cos(psi)*tau
   aa=aa*(3.0/8.0)*0.7*0.9*(RR/pi)
!#endif

!CH debug   aa=sin(delta)*sin(psi)*tau + cos(delta)*cos(psi)*sin(tau)
!CH debug   aa=aa*(3.0/8.0)*(1.-0.7*0.9)*(RR/pi)

   JJ=aa/tau

   k1=kk
   
   A=(Q**2*tau*exp(k1*deep))/(alphalph*JJ) 
   B=(alphalph*JJ*exp(-k1*deep)/Q)**2

   alpha=2.0*A*(sqrt(1.0+B)-1.0)

end function alpha
end module

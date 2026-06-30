module m_get_rhs
contains
subroutine get_rhs(rhs,mem,alpha,k)
   use mod_dimensions
   use mod_fparams
   use mod_states
   implicit none
   type(states), intent(out) :: rhs
   type(states), intent(in) :: mem
   real,    intent(in) :: alpha
   integer, intent(in) :: k

   real fac1,fac2

   fac1=(alpha*mem%N(k))/(jup+mem%N(k)) - r
   fac2=(c*(mem%P(k)-P0))/(Kgraz+mem%P(k)-P0)

   rhs%N(k) = -fac1*mem%P(k) 
   rhs%P(k) = fac1*mem%P(k) - fac2*mem%H(k)
   rhs%H(k) = fff*fac2*mem%H(k) - gg*mem%H(k)

end subroutine get_rhs
end module

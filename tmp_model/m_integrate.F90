module m_integrate
contains
#if defined (PARAM)
subroutine integrate(forw,MM,bnd0,bndH,totdepth,t0,t1,dtref,iens)
#else
subroutine integrate(forw,MM,bnd0,bndH,totdepth,t0,t1,dtref)
#endif
   use mod_dimensions
   use mod_states
   use m_getmxlay
   use m_diffcoeff
   use m_get_rhs
   use m_alpha
#if defined (PARAM)  
   use m_params_assim
#endif  
   implicit none

   type(states), intent(inout) :: forw
   type(local_states), intent(in)    :: bnd0
   type(local_states), intent(in)    :: bndH
   real, intent(in) :: MM(ndim)      
   real, intent(in) :: totdepth
   real, intent(in) :: t0
   real, intent(in) :: t1
   real, intent(in) :: dtref
#if defined (PARAM)  
   integer::iens
#endif   
! automatic arrays
   type(states) new
   type(states) rhs
   type(states) diff   
   real A(kdim)


   real dt
   real dz
   real deep
   real time
   real tmp
   real ppave  ! average phytoplankton concentration above pos z(k)
   real mixlay
   real dz2,dz2i
   type(states) :: f1
   integer k,n,nrt

   dz=(totdepth-0.0)/float(kdim-1)
   dz2i=1.0/(dz*dz)

   rhs=0.0

   A(1)=diffcoeff(0.0,20.0,totdepth)
   
   dt=dz**2/(2.0*A(1))
   dt=dt/1.5
   nrt=int((t1-t0)/dt)+1
   dt=(t1-t0)/float(nrt)

   do n = 1,nrt              
      time=t0+float(n-1)*dt
      mixlay=getmxlay(MM,ndim,time,dtref)
      do k=1,kdim
         deep=float(k-1)*dz
         A(k)=diffcoeff(deep,mixlay,totdepth)
      enddo

      new%N(1) = (1.0/1.5)*(2.0*forw%N(2) - 0.5*forw%N(3) + dz*bnd0%N)
      new%P(1) = (1.0/1.5)*(2.0*forw%P(2) - 0.5*forw%P(3) + dz*bnd0%P)
      new%H(1) = (1.0/1.5)*(2.0*forw%H(2) - 0.5*forw%H(3) + dz*bnd0%H)
   
      do k = 2,kdim-1
         deep=float(k-1)*dz

         diff%N(k)=0.5*dz2i*((A(k+1)+A(k)  )*(forw%N(k+1)-forw%N(k))&
                            -(A(k)  +A(k-1))*(forw%N(k)  -forw%N(k-1)))
         diff%P(k)=0.5*dz2i*((A(k+1)+A(k)  )*(forw%P(k+1)-forw%P(k))&
                            -(A(k)  +A(k-1))*(forw%P(k)  -forw%P(k-1)))
         diff%H(k)=0.5*dz2i*((A(k+1)+A(k)  )*(forw%H(k+1)-forw%H(k))&
                            -(A(k)  +A(k-1))*(forw%H(k)  -forw%H(k-1)))

         ppave=sum(forw%P(1:k))/float(k)
         tmp=alpha(deep,time,ppave)

#if defined (PARAM)
         if (iens.gt.0) then
	   call get_rhs_da(rhs,forw,tmp,k,iens)
	 else
           call get_rhs(rhs,forw,tmp,k)
	 endif
#else
         call get_rhs(rhs,forw,tmp,k)
#endif
         new%P(k)=forw%P(k)+dt*(diff%P(k)+rhs%P(k))
         new%N(k)=forw%N(k)+dt*(diff%N(k)+rhs%N(k))
         new%H(k)=forw%H(k)+dt*(diff%H(k)+rhs%H(k))

      enddo

      new%P(kdim) = bndH%P
      new%N(kdim) = bndH%N
      new%H(kdim) = bndH%H

      forw=new

   enddo
end subroutine integrate

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1
   !ehouarn!
   subroutine integrate_pphys(forw,MM,bnd0,bndH,totdepth,t0,t1,dtref)
   use mod_dimensions
   use mod_states
   use m_getmxlay
   use m_diffcoeff
   use m_get_rhs
   use m_alpha
   use m_random
   implicit none

   type(states), intent(inout) :: forw
   type(local_states), intent(in)    :: bnd0
   type(local_states), intent(in)    :: bndH
   real, intent(in) :: MM(ndim)      
   real, intent(in) :: totdepth
   real, intent(in) :: t0
   real, intent(in) :: t1
   real, intent(in) :: dtref

! automatic arrays
   type(states) new
   type(states) rhs
   type(states) diff   
   real A(kdim)


   real dt
   real dz
   real deep
   real time
   real tmp
   real ppave  ! average phytoplankton concentration above pos z(k)
   real mixlay
   real dz2,dz2i
   type(states) :: f1
   integer k,n,nrt
   real::workb(3)
   
   dz=(totdepth-0.0)/float(kdim-1)
   dz2i=1.0/(dz*dz)

   rhs=0.0

   A(1)=diffcoeff(0.0,20.0,totdepth)
   
   dt=dz**2/(2.0*A(1))
   dt=dt/1.5
   nrt=int((t1-t0)/dt)+1
   dt=(t1-t0)/float(nrt)

   do n = 1,nrt                
      time=t0+float(n-1)*dt
      mixlay=getmxlay(MM,ndim,time,dtref)
      
      !ehouarn!
      call random(workb,3)
      workb(2)=1+0.1*workb(1)
      if(workb(2).gt.0.)then
        mixlay=mixlay*workb(2)     
      endif
      
      do k=1,kdim
         deep=float(k-1)*dz
         A(k)=diffcoeff(deep,mixlay,totdepth)
      enddo

      new%N(1) = (1.0/1.5)*(2.0*forw%N(2) - 0.5*forw%N(3) + dz*bnd0%N)
      new%P(1) = (1.0/1.5)*(2.0*forw%P(2) - 0.5*forw%P(3) + dz*bnd0%P)
      new%H(1) = (1.0/1.5)*(2.0*forw%H(2) - 0.5*forw%H(3) + dz*bnd0%H)

      do k = 2,kdim-1
         deep=float(k-1)*dz

         diff%N(k)=0.5*dz2i*((A(k+1)+A(k)  )*(forw%N(k+1)-forw%N(k))&
                            -(A(k)  +A(k-1))*(forw%N(k)  -forw%N(k-1)))
         diff%P(k)=0.5*dz2i*((A(k+1)+A(k)  )*(forw%P(k+1)-forw%P(k))&
                            -(A(k)  +A(k-1))*(forw%P(k)  -forw%P(k-1)))
         diff%H(k)=0.5*dz2i*((A(k+1)+A(k)  )*(forw%H(k+1)-forw%H(k))&
                            -(A(k)  +A(k-1))*(forw%H(k)  -forw%H(k-1)))

         ppave=sum(forw%P(1:k))/float(k)
         tmp=alpha(deep,time,ppave)

         call get_rhs(rhs,forw,tmp,k)

         new%P(k)=forw%P(k)+dt*(diff%P(k)+rhs%P(k))
         new%N(k)=forw%N(k)+dt*(diff%N(k)+rhs%N(k))
         new%H(k)=forw%H(k)+dt*(diff%H(k)+rhs%H(k))

      enddo

      new%P(kdim) = bndH%P
      new%N(kdim) = bndH%N
      new%H(kdim) = bndH%H

      forw=new

   enddo
end subroutine integrate_pphys

end module

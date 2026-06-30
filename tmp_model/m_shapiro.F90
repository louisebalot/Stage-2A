module m_shapiro
contains
subroutine shapiro(order,y,idim)
   integer, intent(in)    :: order
   integer, intent(in)    :: idim
   real,    intent(inout) :: y(idim)

   integer, parameter :: shdim=16
   real sh(0:shdim)

   call shapiro_ini(order,sh,shdim)
   call shfilt(sh,shdim,order,y,idim)
end subroutine


subroutine shapiro_ini(n,sh,shdim)
   integer, intent(in) :: shdim
   real,    intent(out):: sh(0:shdim)
   integer, intent(in) :: n

   integer i,j
   real f2n,fj,f2nj,ff

   if((n == 1).OR.(n == 2).OR.(n == 4).OR.(n == 8))then
      ff=2.0**(2*n)
      f2n=1

      do j=1,2*n
         f2n=f2n*float(j)
      enddo

      fj=1
      sh(0)=((-1.0)**(n-1))/ff

      do j=1,n
         f2nj=1
         do i=1,2*n-j
            f2nj=f2nj*float(i)
         enddo
         fj=fj*float(j)
         sh(j)=(-1.0)**(n+1-j)*f2n/(ff*fj*f2nj)
      enddo
   else
      write(*,*)'error in shfact.  n=',n
      stop
   endif
end subroutine shapiro_ini

subroutine shfilt(sh,shdim,ish,y,yn)
   integer, intent(in) :: shdim
   real,    intent(in) :: sh(0:shdim)
   integer, intent(in) :: ish
   integer, intent(in) :: yn
   real,    intent(out):: y(yn)
   integer i,j,m

   real, allocatable :: z(:)

   allocate(z(yn))

   z=y

   do m=2,ish
     do j=0,ish-1
       mm=1-(m-ish+j)
       if(mm.GE.0)then
         z(m)=z(m)+sh(j)*(2.0*y(1)-y(1+mm)+y(m+ish-j))
       else
         z(m)=z(m)+sh(j)*(y(m+ish-j)+y(m-ish+j))
       endif
     enddo
     z(m)=(sh(ish))*y(m)+z(m)
   enddo

   do m=ish+1,yn-ish
     do j=0,ish-1
         z(m)=z(m)+sh(j)*(y(m+ish-j)+y(m-ish+j))
     enddo
     z(m)=(sh(ish))*y(m)+z(m)
   enddo

   do m=yn-ish+1,yn-1
     do j=0,ish-1
       mm=(m+ish-j)-yn
       if(mm.GE.0)then
         z(m)=z(m)+sh(j)*(2.0*y(yn)-y(yn-mm)+y(m-ish+j))
       else
         z(m)=z(m)+sh(j)*(y(m+ish-j)+y(m-ish+j))
       endif
     enddo
     z(m)=(sh(ish))*y(m)+z(m)
   enddo

   y=z

   deallocate(z)

end subroutine shfilt

end module

module mod_fparams
#if defined (PARONLY)
   real,save:: jup=0.5          ! uptake half saturation
   real ,save:: r=0.07!0.1!r!0.07           ! plant metabolic loss
   real,save :: c=1.0            ! max grazing rate
   real,save :: P0=0.1           ! grazing threshold
   real,save :: Kgraz=1.0        ! grazing half saturation
   real,save :: fff=0.5!0.7!0.5!fff!0.5          ! grazing efficiency
   real,save :: gg=0.07!0.1!gg!0.07          ! loss to carnivores
#else   
   real, parameter :: jup=0.5          ! uptake half saturation
   real, parameter :: r=0.07!0.1!r!0.07           ! plant metabolic loss
   real, parameter :: c=1.0            ! max grazing rate
   real, parameter :: P0=0.1           ! grazing threshold
   real, parameter :: Kgraz=1.0        ! grazing half saturation
   real, parameter :: fff=0.5!0.7!0.5!fff!0.5          ! grazing efficiency
   real, parameter :: gg=0.07!0.1!gg!0.07          ! loss to carnivores
#endif  
end module mod_fparams


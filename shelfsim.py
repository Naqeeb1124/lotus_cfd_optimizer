import numpy as np
import matplotlib.pyplot as plt




N_shelves = 10                  
time_total = 5                  
dt = 0.05                       
steps = int(time_total / dt)


T_inlet = 45.0                  
Y_inlet = 0.008                 
m_dot_air = 0.015               
Cp_air = 1006.0                 


m_pollen_per_shelf = 1.5        
M_initial = 0.40                
M_eq = 0.05                     
lambda_vap = 2.26e6             


k_drying = 0.35                 




M_pollen = np.zeros((steps, N_shelves))
T_air = np.zeros((steps, N_shelves))
RH_air = np.zeros((steps, N_shelves))

M_pollen[0, :] = M_initial




def calc_saturation_pressure(T_celsius):
    return 1000 * np.exp(16.3872 - (3885.7 / (T_celsius + 230.170)))

for t in range(steps - 1):
    T_current = T_inlet
    Y_current = Y_inlet
    air_is_saturated = False
    
    for i in range(N_shelves):
        if air_is_saturated:
            
            M_pollen[t+1, i] = M_pollen[t, i]
            T_air[t, i] = T_current
            RH_air[t, i] = 100.0
            continue
            
        
        dM_dt = -k_drying * (M_pollen[t, i] - M_eq)
        
        if M_pollen[t, i] <= M_eq: 
            dM_dt = 0
            
        
        M_pollen[t+1, i] = M_pollen[t, i] + dM_dt * dt
        
        
        water_evap_rate = abs(dM_dt) * m_pollen_per_shelf / 3600.0
        
        
        T_drop = (water_evap_rate * lambda_vap) / (m_dot_air * Cp_air)
        T_next = T_current - T_drop
        
        
        Y_next = Y_current + (water_evap_rate / m_dot_air)
        
        
        P_sat = calc_saturation_pressure(T_next)
        P_atm = 101325.0
        P_vapor = (Y_next * P_atm) / (0.622 + Y_next)
        RH = (P_vapor / P_sat) * 100.0
        
        if RH >= 98.0:
            RH = 100.0
            air_is_saturated = True
            
        
        T_air[t, i] = T_next
        RH_air[t, i] = RH
        
        
        T_current = T_next
        Y_current = Y_next




plt.style.use('dark_background')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
shelf_axis = np.arange(1, N_shelves + 1)


eval_time = int(1.5 / dt) 


ax1.plot(shelf_axis, T_air[eval_time, :], 'r-o', linewidth=2)
ax1.set_title('Air Temperature Plunge Across Shelves')
ax1.set_xlabel('Shelf Number (1=Bottom, 10=Top)')
ax1.set_ylabel('Temperature (°C)')
ax1.grid(True, alpha=0.2)


ax2.plot(shelf_axis, RH_air[eval_time, :], 'b-o', linewidth=2)
ax2.set_title('Relative Humidity Explosion')
ax2.set_xlabel('Shelf Number')
ax2.set_ylabel('Relative Humidity (%)')
ax2.axhline(y=100, color='red', linestyle='--', label='Saturation Limit')
ax2.grid(True, alpha=0.2)
ax2.legend()


ax3.plot(shelf_axis, M_pollen[eval_time, :] * 100, 'g-o', linewidth=2)
ax3.set_title('Pollen Moisture Content (at t = 1.5 hrs)')
ax3.set_xlabel('Shelf Number')
ax3.set_ylabel('Moisture Content (%)')
ax3.grid(True, alpha=0.2)

plt.tight_layout()
plt.show()
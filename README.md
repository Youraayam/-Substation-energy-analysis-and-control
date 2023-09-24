# -Substation-energy-analysis-and-control
This was the lab exercise where students were analyzing Energy of the substation rig based on the measurements; - Sizing of a PID controller; - Analysis of control setting for a PID controller; - Analysis of time delay in control
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 22:40:02 2023

@author: aayam
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 21:33:22 2023
aa
@author: aayam
"""
import numpy
import math 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import r2_score
from matplotlib.sankey import Sankey

 


assignment_2 = pd.read_csv('assignment_2.csv',encoding='unicode_escape')
assignment_2.info()
assignment_2.head()
assignment_2["Time"] = pd.to_datetime(assignment_2["Time"])
print(assignment_2.columns.tolist())

#Task1.1
Times = assignment_2["Time"]
cp = 4.187


#Hot water
P_HW_t = assignment_2["EO1Power"]
#1Heater
#(secondary) 
flowrate_H1_S = assignment_2["F210"]/3600
temp_H1_S_h =assignment_2["T113"]
temp_H1_S_l =assignment_2["T162"]
Power_H1_s = flowrate_H1_S * cp * (temp_H1_S_h - temp_H1_S_l)

#1Heater 
#(primary)
flowrate_H1_p = assignment_2["F202"]/3600
temp_H1_p_h =assignment_2["T1"]
temp_H1_p_l =assignment_2["T132"]
Power_H1_p = flowrate_H1_p * cp * (temp_H1_p_h - temp_H1_p_l)


plt.figure().set_figwidth(15)
plt.plot(Times, Power_H1_s,'--b',Times, Power_H1_p,'--r')
plt.title("Power needed for hot water heat exchanger (heater)")
plt.legend(['Secondary side', 'Primary side'])
plt.xlabel("Time(Day|Hour|minute)")
plt.ylabel("Power (KW)")
plt.xticks(rotation=45)
plt.grid()
plt.show()

#d_ta = assignment_2["T1"] - assignment_2["T162"]
#d_tb = assignment_2["T132"] - assignment_2["T113"]

#LMTD = (d_ta - d_tb)/(math.log(d_ta/d_tb))



#2Heater
#(secondary) 
#Power_H2_s = P_HW_t - Power_H1_s

#1Heater 
#(primary)
#Power_H2_p = Power_H2_s * eff 

Power_H2_p1 = []
Power_H2_p2 = []
Power_H2_s1 = []
Power_H2_s2 = []
x11 = []
x22 = []
O = 7
k = 7
b = numpy.zeros(O)
b[0] = 0 ; b[1] = 1; b[2] = 2; b[3] = 3 ; b[4] = 4; b[5] = 5; b[6] = 6
k = numpy.zeros(O)
k[0] = 0.0 ; b[1] = 0.2; b[2] = 0.4; b[3] = 0.6 ; b[4] = 0.8; b[5] = 1



for i in range(0,7,1):
    
    NTU =  b[i]
    
    for j in range (0,7,1):
        
        R = k[j]
        
        if R in range (0,1):
            a = numpy.exp(-NTU*(1-R))
            effa1 = ((1- a)/(1- R * a))
            temp_H2_p_h1 = (effa1 * assignment_2["T147"] - assignment_2["T153"])/(effa1-1)
            Power1 = ((assignment_2["FEO1"] * cp * (temp_H2_p_h1 - assignment_2["TEO1ut"])))/3600
            Power11 = effa1*Power1
            x1 = Times
            Power_H2_p1.append(Power1)
            Power_H2_s1.append(Power11)
            x11.append(x1)
            

        elif R in range(1, 2):
            
            effa2 = (NTU)/(1+NTU)
            temp_H2_p_h2 = (effa2 * assignment_2["T147"] - assignment_2["T153"])/(effa2-1)
            Power2 = (assignment_2["FEO1"] * cp * (temp_H2_p_h2 - assignment_2["TEO1ut"]))/3600
            Power22 = effa1*Power2
            Power_H2_p2.append(Power2)
            Power_H2_s2.append(Power22)
            x2 = Times
            x22.append(x2)
        else: 
            
            print ("Your number wasn't in the correct range")

    plt.figure().set_figwidth(15)
    plt.plot(x11, Power_H2_p1,'--r',color='red')
    plt.plot(x11, Power_H2_s1,'--g',color='green')
    plt.xticks(rotation=45)
    plt.title("Power needed for hot water heat exchanger (preheater)")
    pop_a = mpatches.Patch(color='red', label='Primary side')
    pop_b = mpatches.Patch(color='green', label='Secondary side')
    plt.legend(handles=[pop_a,pop_b])
    plt.xlabel("Time(Day|Hour|minute)")
    plt.ylabel("Power (KW)")
    plt.grid()
    plt.show()
    
    plt.figure().set_figwidth(15)
    plt.plot(x22, Power_H2_p2,'--b')
    plt.plot(x22, Power_H2_s2,'--p')
    plt.xticks(rotation=45)
    plt.title("Power needed for hot water heat exchanger (preheater1)")
    plt.legend(['Primary side', 'Secondary side'])
    plt.xlabel("Time(Day|Hour|minute)")
    plt.ylabel("Power (KW)")
    plt.grid()
    plt.show()        
            


#Space heating
P_SH_main = assignment_2["EO5Power"]
P_SH_vent = assignment_2["EO8Power"]
P_SH_rad = assignment_2["EO7Power"]

plt.figure().set_figwidth(15)
plt.plot(Times, P_SH_main,'--b',Times, P_SH_vent,'r', Times, P_SH_rad, '--g')
plt.title("Power needed for space heating")
plt.legend(['EO5Power(total)', 'EO8PPower(vent)','EO7PPower(rad)'])
plt.xlabel("Time(Day|Hour|minute)")
plt.ylabel("Power (KW)")
plt.xticks(rotation=45)
plt.grid()
plt.show()


#Task 1.2 

#Main heat exchanger 

M_temp_h = assignment_2["TEO5in"]
M_temp_l = assignment_2["TEO5ut"]

M_flow = assignment_2.loc[:,"FEO5"]

M_power =( M_flow * cp * (M_temp_h-M_temp_l))/3600

plt.figure().set_figwidth(15)
plt.plot(Times, P_SH_main,'--b',Times, M_power,'r')
plt.title("Comparison between measured power and calculated power for main heat exchanger at space heating")
plt.legend(['Measured', 'Calculated'])
plt.xlabel("Time(Day|Hour|minute)")
plt.ylabel("Power (KW)")
plt.xticks(rotation=45)
plt.grid()
plt.show()



plt.figure().set_figwidth(15)
a, b = numpy.polyfit(P_SH_main,M_power,1)
plt.scatter(P_SH_main,M_power)
plt.plot(P_SH_main,a*P_SH_main+b)
plt.title("Comparison between measured power and calculated power for main heat exchanger at space heating")
plt.xlabel("Measured Power(KW)")
plt.ylabel("Calculated Power(KW)")
plt.xticks(rotation=45)
corr_matrix1 = numpy.corrcoef(P_SH_main,M_power)
corr1 = corr_matrix1[0,1]
R_sq1 = corr1**2

print(R_sq1)
plt.annotate("r-squared = {:.3f}".format(R_sq1), (0, 13))

plt.grid()
plt.show()



#Ventilation

V_temp_h = assignment_2["Trin1"]
V_temp_l = assignment_2["TEO8ut"]

V_flow = assignment_2["F206"]

V_power =( V_flow * cp * (V_temp_h-V_temp_l))/3600
plt.figure().set_figwidth(15)
plt.plot(Times, P_SH_vent,'--b',Times, V_power,'r')
plt.title("Comparison between measured power and calculated power for ventilation at space heating")
plt.legend(['Measured ', 'Calculated '])
plt.xlabel("Time(Day|Hour|minute)")
plt.ylabel("Power (KW)")
plt.xticks(rotation=45)
plt.grid()
plt.show()

plt.figure().set_figwidth(15)
plt.scatter(P_SH_vent,V_power)
c, d = numpy.polyfit(P_SH_vent,V_power,1)
plt.plot(P_SH_vent,c*P_SH_vent+d)
corr_matrix2 = numpy.corrcoef(P_SH_vent,V_power)
corr2 = corr_matrix2[0,1]
R_sq2 = corr2**2
print(R_sq2)
plt.annotate("r-squared = {:.3f}".format(R_sq2), (1, 5))
plt.title("Comparison between measured power and calculated power for ventilation at space heating")
plt.xlabel("Measured Power(KW)")
plt.ylabel("Calculated Power(KW)")
plt.xticks(rotation=45)
plt.grid()
plt.show()




#Radiator 

R_temp_h = assignment_2["Trin1"]
R_temp_l = assignment_2["TEO7ut"]

R_flow = assignment_2["FEO4"] - assignment_2["F206"]

R_power =( R_flow * cp * (R_temp_h-R_temp_l))/3600
plt.figure().set_figwidth(15)
plt.plot(Times, P_SH_rad,'--b',Times, R_power,'r')
plt.title("Comparison between measured power and calculated power for radiator at space heating")
plt.legend(['Measured', 'Calculated'])
plt.xlabel("Time(Day|Hour|minute)")
plt.ylabel("Power (KW)")
plt.xticks(rotation=45)
plt.grid()
plt.show()

plt.figure().set_figwidth(15)
plt.scatter(P_SH_rad,R_power)
e, f = numpy.polyfit(P_SH_rad,R_power,1)
plt.plot(P_SH_rad,e*P_SH_rad+f)
corr_matrix3 = numpy.corrcoef(P_SH_rad,R_power)
corr3 = corr_matrix3[0,1]
R_sq3 = corr3**2
print(R_sq3)
plt.annotate("r-squared = {:.3f}".format(R_sq3), (8, 0))
plt.title("Comparison between measured power and calculated power for radiator at space heating")
plt.xlabel("Measured Power(KW)")
plt.ylabel("Calculated Power(KW)")
plt.xticks(rotation=45)
plt.grid()
plt.show()


#Calculating energy 
#1.Supply
Sup_energy = (((assignment_2['FEO1']* cp * (assignment_2['TEO1in'] -assignment_2['TEO1ut']))/3600)/3600).sum()
print(Sup_energy)
Tap_energy1 = ((Power_H1_p) /3600).sum()
print(Tap_energy1)
Main_he_energy = (M_power/3600).sum()
print(Main_he_energy)
Rad_energy = (R_power /3600).sum()
print(Rad_energy)
Ven_energy = (V_power/3600).sum()
print(Ven_energy)
Tap_energy2 =(Sup_energy-(Tap_energy1+Rad_energy+Ven_energy))
print(Tap_energy2)


#1.3 Energy Balance 

#Primary side
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(2,1,1)
ax.axis('off')
sankey = Sankey(ax = ax, scale=0.02, offset=0.3)
sankey.add(flows = [8.8, -4.3, -4.5],
                    labels = ['Total(KWh)', 'Tap Water(KWh)', 'Space heating(KWh)'],
                    orientations = [0,0,1]
)
sankey.add(flows = [4.3, -1.94, -2.36],
                    labels=['Tap Water(KWh)', 'Primary(KWh)', 'Preheater(KWh)'],
                    orientations=[0,-1,0],
                    prior = 0,
                    connect = (1, 0)
)
sankey.add(flows = [4.5, -1.8, -2.7], 
                    labels=['Space heating(KWh)', 'Radiator(KWh)', 'Ventilation(KWh)'],
                    orientations=[1,0,-1],
                    prior = 0,
                    connect = (2, 0)
)  
diagrams = sankey.finish()
plt.legend(loc='lower right')
plt.title('Energy Balance of substation primary side')
plt.show()


#Secondary side 

sup = Sup_energy
print(sup)

P1 =  Power_H1_s
Tap_sec1 = (P1/3600).sum() 
print(Tap_sec1)
Tap_loss1 = Tap_energy1 - Tap_sec1
print(Tap_loss1)


Sh_sec = ((M_power)/3600).sum() 
Sh_loss = ((M_power)/3600).sum() - (Rad_energy+Ven_energy)
print(Sh_sec)
print(Sh_loss)

P2 =  assignment_2['F210']*cp*(assignment_2['T153']-assignment_2['T147'])
Tap_sec2 = ((P2/3600)/3600).sum() 
print(Tap_sec2)
Tap_loss2 = ((P2/3600)/3600).sum() *0.005
print(Tap_loss2)


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(2,1,1)
ax.axis('off')
sankey = Sankey(ax = ax, scale=0.02, offset=0.3)
sankey.add(flows = [8.8,-4.23,-4.43,-0.14],
                    labels = ['Total(KWh)','Tap water(KWh)', 'Space heating (KWh)','other losses'],
                    orientations = [0,0,1,-1]
)
sankey.add(flows =  [4.23, -1, -0.91,-1.2,-1.12],
                    labels=['Tap Water(KWh)','Primary(KWh)','Primary(KWh)loss','Preheating(KWh)','Preheating(KWh)loss'],
                    orientations=[0,0,1,1,-1],
                    prior = 0,
                    connect = (1, 0)
)
sankey.add(flows = [4.43, -4.21, -0.22], 
                    labels=['Space heating(KWh)', 'Space heating(KWh)gain', 'Space heating(KWh) loss'],
                    orientations=[1,0,-1],
                    prior = 0,
                    connect = (2, 0)
)  
diagrams = sankey.finish()
plt.legend(loc='lower right')
plt.title('Energy Balance of substation secondary side')
plt.show()

#1.4

heat_tap = (assignment_2['F210']*cp*(assignment_2['T113']-assignment_2['T147']))/3600
plt.figure().set_figwidth(15)
plt.plot(Times, heat_tap,'--b')
plt.title("Power needed for hot water(total delivered_")
plt.legend(['primary'])
plt.xlabel("Time(Day|Hour|minute)")
plt.ylabel("Power (KW)")
plt.xticks(rotation=45)
x_axis = assignment_2['x']
area_tap = numpy.trapz(heat_tap, x_axis , dx =1)/3600
print(area_tap)
plt.annotate("Energy used = {:.3f}".format(area_tap), (1, 15))
plt.grid()
plt.show()



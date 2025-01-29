import numpy as np
import matplotlib.pyplot as plt
import support_files_car_general as sfc_g
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
from qpsolvers import *
np.set_printoptions(suppress=True)

import platform
print("Python " + platform.python_version())
import numpy as np
print("Numpy " + np.__version__)
import matplotlib
print("Matplotlib " + matplotlib.__version__)

# Create an object for the support functions.
support=sfc_g.SupportFilesCar()
constants=support.constants

# Load the constant values needed in the main file
Ts=constants['Ts']
outputs=constants['outputs'] # number of outputs (psi, Y)
hz = constants['hz'] # horizon prediction period
time_length=constants['time_length'] # duration of the manoeuvre
inputs=constants['inputs']
x_lim=constants['x_lim']
y_lim=constants['y_lim']
trajectory=constants['trajectory']

# Generate the refence signals
t=np.zeros((int(time_length/Ts+1)))

for i in range(1,len(t)):
    t[i]=np.round(t[i-1]+Ts,2)

x_dot_ref,y_dot_ref,psi_ref,X_ref,Y_ref=support.trajectory_generator(t)
sim_length=len(t) # Number of control loop iterations
refSignals=np.zeros(len(X_ref)*outputs)

# Build up the reference signal vector:
# refSignal = [x_dot_ref_0, psi_ref_0, X_ref_0, Y_ref_0, x_dot_ref_1, psi_ref_1, X_ref_1, Y_ref_1, x_dot_ref_2, psi_ref_2, X_ref_2, Y_ref_2, ... etc.]
k=0
for i in range(0,len(refSignals),outputs):
    refSignals[i]=x_dot_ref[k]
    refSignals[i+1]=psi_ref[k]
    refSignals[i+2]=X_ref[k]
    refSignals[i+3]=Y_ref[k]
    k=k+1

# Load the initial states
# If you want to put numbers here, please make sure that they are float and not
# integers. It means that you should add a point there.
# Example: Please write 0. in stead of 0 (Please add the point to make it float)
x_dot=x_dot_ref[0]
y_dot=y_dot_ref[0]
psi=psi_ref[0]
psi_dot=0.
X=X_ref[0]
Y=Y_ref[0]

states=np.array([x_dot,y_dot,psi,psi_dot,X,Y])
statesTotal=np.zeros((len(t),len(states))) # It will keep track of all your states during the entire manoeuvre
statesTotal[0][0:len(states)]=states

######################### Accelerations ########################################
x_dot_dot=0.
y_dot_dot=0.
psi_dot_dot=0.

accelerations=np.array([x_dot_dot,y_dot_dot,psi_dot_dot])
accelerations_total=np.zeros((len(t),len(accelerations)))

################################################################################
# X_opt_total=np.zeros((len(t),hz))
# Y_opt_total=np.zeros((len(t),hz))

# Load the initial input
U1=0 # Input at t = -0.02 s (steering wheel angle in rad (delta))
U2=0 # Input at t = -0.02 s (acceleration in m/s^2 (a))
UTotal=np.zeros((len(t),2)) # To keep track all your inputs over time
UTotal[0][0]=U1
UTotal[0][1]=U2

# Initiate the controller - simulation loops
k=0
du=np.zeros((inputs*hz,1))

# # To extract X_opt from predicted x_aug_opt
# C_X_opt=np.zeros((hz,(len(states)+np.size(U1)+np.size(U2))*hz))
# for i in range(4,hz+4):
#     C_X_opt[i-4][i+7*(i-4)]=1
#
# # To extract Y_opt from predicted x_aug_opt
# C_Y_opt=np.zeros((hz,(len(states)+np.size(U1)+np.size(U2))*hz))
# for i in range(5,hz+5):
#     C_Y_opt[i-5][i+7*(i-5)]=1

# Arrays for the animation - every 5th state goes in there (once in 0.1 seconds, because Ts=0.02 seconds)
t_ani=[]
x_dot_ani=[]
psi_ani=[]
X_ani=[]
Y_ani=[]
delta_ani=[]

for i in range(0,sim_length-1):

    # Generate the discrete state space matrices
    Ad,Bd,Cd,Dd=support.state_space(states,U1,U2)

    # Generate the augmented current state and the reference vector
    x_aug_t=np.transpose([np.concatenate((states,[U1,U2]),axis=0)])

    # From the refSignals vector, only extract the reference values from your [current sample (NOW) + Ts] to [NOW+horizon period (hz)]
    # Example: Ts=0.1 seconds, t_now is 3 seconds, hz = 15 samples, so from refSignals vectors, you move the elements to vector r:
    # r=[x_dot_ref_3.1, psi_ref_3.1, X_ref_3.1, Y_ref_3.1, x_dot_ref_3.2, psi_ref_3.2, X_ref_3.2, Y_ref_3.2, ... , x_dot_ref_4.5, psi_ref_4.5, X_ref_4.5, Y_ref_4.5]
    # With each loop, it all shifts by 0.1 second because Ts=0.1 s
    k=k+outputs
    if k+outputs*hz<=len(refSignals):
        r=refSignals[k:k+outputs*hz]
    else:
        r=refSignals[k:len(refSignals)]
        hz=hz-1

    # Generate the compact simplification matrices for the cost function
    Hdb,Fdbt,Cdb,Adc,G,ht=support.mpc_simplification(Ad,Bd,Cd,Dd,hz,x_aug_t,du)
    ft=np.matmul(np.concatenate((np.transpose(x_aug_t)[0][0:len(x_aug_t)],r),axis=0),Fdbt)
    # Hdb=np.array([[10.,1.],[1.,10.]])
    # ft=np.array([0.,0.])
    # G=np.array([[-1.,0.],[0.,-1.],[-1.4,-1.]])
    # ht=np.array([-0.05,-0.07,-0.15])
    ######################### Constraints #####################################
    try:
        du=solve_qp(Hdb,ft,G,ht,solver="cvxopt")
        du=np.transpose([du])
        # print(du)
        # exit()
    except ValueError as ve:
        print(Hdb)
        print(ft)
        print(G)
        print(ht)
        print(Adc)
        print(x_aug_t)
        print(du)
        print(i)
        break;

    #############################################################################

    # # No constraints
    # du=-np.matmul(np.linalg.inv(Hdb),np.transpose([ft]))

    # # Predicted states
    # x_aug_opt=np.matmul(Cdb,du)+np.matmul(Adc,x_aug_t)
    # X_opt=np.matmul(C_X_opt[0:hz,0:(len(states)+np.size(U1)+np.size(U2))*hz],x_aug_opt)
    # Y_opt=np.matmul(C_Y_opt[0:hz,0:(len(states)+np.size(U1)+np.size(U2))*hz],x_aug_opt)
    #
    # X_opt=np.transpose((X_opt))[0]
    # X_opt_total[i+1][0:hz]=X_opt
    # Y_opt=np.transpose((Y_opt))[0]
    # Y_opt_total[i+1][0:hz]=Y_opt

    # Update the real inputs
    U1=U1+du[0][0]
    U2=U2+du[1][0]

    UTotal[i+1][0]=U1
    UTotal[i+1][1]=U2

    states,x_dot_dot,y_dot_dot,psi_dot_dot=support.open_loop_new_states(states,U1,U2)
    statesTotal[i+1][0:len(states)]=states

    ######################### Accelerations ####################################
    accelerations=np.array([x_dot_dot,y_dot_dot,psi_dot_dot])
    accelerations_total[i+1][0:len(accelerations)]=accelerations

    # This is to monitor the progress of the simulation
    if i%500==0:
        print("Progress: "+str(round(i/sim_length*100,2))+"%")

    # To make the animations 5x faster
    if i%5==1:
        t_ani=np.concatenate([t_ani,[t[i]]])
        x_dot_ani=np.concatenate([x_dot_ani,[states[0]]])
        psi_ani=np.concatenate([psi_ani,[states[2]]])
        X_ani=np.concatenate([X_ani,[states[4]]])
        Y_ani=np.concatenate([Y_ani,[states[5]]])
        delta_ani=np.concatenate([delta_ani,[U1]])

################################ ANIMATION LOOP ###############################

frame_amount=len(X_ani)
lf=constants['lf']
lr=constants['lr']

def update_plot(num):
    hz=constants['hz']
    car_1.set_data([X_ani[num]-lr*np.cos(psi_ani[num]),X_ani[num]+lf*np.cos(psi_ani[num])],
        [Y_ani[num]-lr*np.sin(psi_ani[num]),Y_ani[num]+lf*np.sin(psi_ani[num])])

    car_determined.set_data(X_ani[0:num],Y_ani[0:num])
    x_dot.set_data(t_ani[0:num],x_dot_ani[0:num])
    yaw_angle.set_data(t_ani[0:num],psi_ani[0:num])
    X_position.set_data(t_ani[0:num],X_ani[0:num])
    Y_position.set_data(t_ani[0:num],Y_ani[0:num])
    # if num<len(X_ani)-5:
    #     car_predicted.set_data(X_opt_total[(num+1)+4*num][0:hz],Y_opt_total[(num+1)+4*num][0:hz])
    # else:
    #     car_predicted.set_data(X_opt_total[(num+1)+4*num][0:0],Y_opt_total[(num+1)+4*num][0:0])

    car_1_body.set_data([-lr*np.cos(psi_ani[num]),lf*np.cos(psi_ani[num])],
        [-lr*np.sin(psi_ani[num]),lf*np.sin(psi_ani[num])])

    car_1_body_extension.set_data([0,(lf+40)*np.cos(psi_ani[num])],
        [0,(lf+40)*np.sin(psi_ani[num])])

    car_1_back_wheel.set_data([-(lr+0.5)*np.cos(psi_ani[num]),-(lr-0.5)*np.cos(psi_ani[num])],
        [-(lr+0.5)*np.sin(psi_ani[num]),-(lr-0.5)*np.sin(psi_ani[num])])

    car_1_front_wheel.set_data([lf*np.cos(psi_ani[num])-0.5*np.cos(psi_ani[num]+delta_ani[num]),lf*np.cos(psi_ani[num])+0.5*np.cos(psi_ani[num]+delta_ani[num])],
        [lf*np.sin(psi_ani[num])-0.5*np.sin(psi_ani[num]+delta_ani[num]),lf*np.sin(psi_ani[num])+0.5*np.sin(psi_ani[num]+delta_ani[num])])

    car_1_front_wheel_extension.set_data([lf*np.cos(psi_ani[num]),lf*np.cos(psi_ani[num])+(0.5+40)*np.cos(psi_ani[num]+delta_ani[num])],
        [lf*np.sin(psi_ani[num]),lf*np.sin(psi_ani[num])+(0.5+40)*np.sin(psi_ani[num]+delta_ani[num])])

    yaw_angle_text.set_text(str(round(psi_ani[num],2))+' rad')
    steer_angle.set_text(str(round(delta_ani[num],2))+' rad')
    body_x_velocity.set_text(str(round(x_dot_ani[num],2))+' m/s')

    return car_determined,car_1,x_dot,yaw_angle,X_position,Y_position,\
    car_1_body,car_1_body_extension,car_1_back_wheel,car_1_front_wheel,car_1_front_wheel_extension,\
    yaw_angle_text,steer_angle,body_x_velocity#,car_predicted


# Set up your figure properties
fig_x=16
fig_y=9
fig=plt.figure(figsize=(fig_x,fig_y),dpi=120,facecolor=(0.8,0.8,0.8))
n=12
m=12
gs=gridspec.GridSpec(n,m)

# Main trajectory
plt.subplots_adjust(left=0.05,bottom=0.08,right=0.95,top=0.95,wspace=0.15,hspace=0)

ax0=fig.add_subplot(gs[:,0:9],facecolor=(0.9,0.9,0.9))
ax0.grid(True)
plt.title('Autonomous vehicle animation (5x faster than the reality)',size=15)
plt.xlabel('X-position [m]',fontsize=15)
plt.ylabel('Y-position [m]',fontsize=15)

# Plot the reference trajectory
ref_trajectory=ax0.plot(X_ref,Y_ref,'b',linewidth=1)

# Draw a motorcycle
car_1,=ax0.plot([],[],'k',linewidth=3)
# car_predicted,=ax0.plot([],[],'-m',linewidth=2)
car_determined,=ax0.plot([],[],'-r',linewidth=1)

# Zoomed vehicle
if trajectory==1:
    ax1=fig.add_subplot(gs[0:6,0:5],facecolor=(0.9,0.9,0.9))
elif trajectory==2:
    ax1=fig.add_subplot(gs[3:9,2:7],facecolor=(0.9,0.9,0.9))
else:
    ax1=fig.add_subplot(gs[2:6,2:5],facecolor=(0.9,0.9,0.9))
ax1.axes.get_xaxis().set_visible(False)
ax1.axes.get_yaxis().set_visible(False)

bbox_props_x_dot=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='b',lw=1.0)
bbox_props_steer_angle=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='r',lw=1.0)
bbox_props_angle=dict(boxstyle='square',fc=(0.9,0.9,0.9),ec='k',lw=1.0)

neutral_line=ax1.plot([-50,50],[0,0],'k',linewidth=1)
car_1_body,=ax1.plot([],[],'k',linewidth=3)
car_1_body_extension,=ax1.plot([],[],'--k',linewidth=1)
car_1_back_wheel,=ax1.plot([],[],'r',linewidth=4)
car_1_front_wheel,=ax1.plot([],[],'r',linewidth=4)
car_1_front_wheel_extension,=ax1.plot([],[],'--r',linewidth=1)


plt.xlim(-5,5)
plt.ylim(-4,4)

body_x_velocity=ax1.text(3,-1.5,'',size='10',color='b',bbox=bbox_props_x_dot)
steer_angle=ax1.text(3,-2.5,'',size='10',color='r',bbox=bbox_props_steer_angle)
yaw_angle_text=ax1.text(3,-3.5,'',size='10',color='k',bbox=bbox_props_angle)

body_x_velocity_word=ax1.text(3.7,3.4,'x_dot',size='10',color='b',bbox=bbox_props_x_dot)
steer_angle_word=ax1.text(3.8,2.5,'delta',size='10',color='r',bbox=bbox_props_steer_angle)
yaw_angle_word=ax1.text(4.2,1.6,'Psi',size='10',color='k',bbox=bbox_props_angle)


# x_dot function
ax2=fig.add_subplot(gs[0:3,9:12],facecolor=(0.9,0.9,0.9))
x_dot_reference=ax2.plot(t,x_dot_ref,'-b',linewidth=1)
x_dot,=ax2.plot([],[],'-r',linewidth=1)
ax2.spines['bottom'].set_position(('data',-9999999))
ax2.yaxis.tick_right()
ax2.grid(True)
plt.xlabel('time [s]',fontsize=15)
plt.ylabel('x_dot [m/s]',fontsize=15)
ax2.yaxis.set_label_position("right")

# Psi function
ax3=fig.add_subplot(gs[3:6,9:12],facecolor=(0.9,0.9,0.9))
yaw_angle_reference=ax3.plot(t,psi_ref,'-b',linewidth=1)
yaw_angle,=ax3.plot([],[],'-r',linewidth=1)
ax3.spines['bottom'].set_position(('data',-9999999))
ax3.yaxis.tick_right()
ax3.grid(True)
plt.xlabel('time [s]',fontsize=15)
plt.ylabel('Psi [rad]',fontsize=15)
ax3.yaxis.set_label_position("right")

# X function
ax4=fig.add_subplot(gs[6:9,9:12],facecolor=(0.9,0.9,0.9))
X_position_reference=ax4.plot(t,X_ref,'-b',linewidth=1)
X_position,=ax4.plot([],[],'-r',linewidth=1)
ax4.spines['bottom'].set_position(('data',-9999999))
ax4.yaxis.tick_right()
ax4.grid(True)
plt.xlabel('time [s]',fontsize=15)
plt.ylabel('X-position [m]',fontsize=15)
ax4.yaxis.set_label_position("right")

# Y function
ax5=fig.add_subplot(gs[9:12,9:12],facecolor=(0.9,0.9,0.9))
Y_position_reference=ax5.plot(t,Y_ref,'-b',linewidth=1)
Y_position,=ax5.plot([],[],'-r',linewidth=1)
ax5.yaxis.tick_right()
ax5.grid(True)
plt.xlabel('time [s]',fontsize=15)
plt.ylabel('Y-position [m]',fontsize=15)
ax5.yaxis.set_label_position("right")


car_ani=animation.FuncAnimation(fig, update_plot,
    frames=frame_amount,interval=20,repeat=True,blit=True)
plt.show()

# # This is to record a video on the animation
# # Matplotlib 3.3.3 needed - close the animation itself to start the recording process
# Writer=animation.writers['ffmpeg']
# writer=Writer(fps=30,metadata={'artist': 'Me'},bitrate=1800)
# car_ani.save('car_mpc_demo_traj3_v2.mp4',writer)
##################### END OF THE ANIMATION ############################



# Plot the world
plt.plot(X_ref,Y_ref,'--b',linewidth=2,label='The trajectory')
plt.plot(statesTotal[:,4],statesTotal[:,5],'r',linewidth=1,label='Car position')
plt.xlabel('X-position [m]',fontsize=15)
plt.ylabel('Y-position [m]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')
plt.xlim(0,x_lim)
plt.ylim(0,y_lim)
plt.xticks(np.arange(0,x_lim+1,int(x_lim/10)))
plt.yticks(np.arange(0,y_lim+1,int(y_lim/10)))
plt.show()


plt.subplot(2,1,1)
plt.plot(t,UTotal[:,0],'r',linewidth=1,label='steering wheel angle')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('steering wheel angle [rad]',fontsize=15)
plt.grid(True)
plt.legend(loc='lower right',fontsize='small')

plt.subplot(2,1,2)
plt.plot(t,UTotal[:,1],'r',linewidth=1,label='applied acceleration')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('applied acceleration [m/s^2]',fontsize=15)
plt.grid(True)
plt.legend(loc='lower right',fontsize='small')
plt.show()



plt.subplot(3,1,1)
plt.plot(t,psi_ref,'--b',linewidth=2,label='Yaw_ref angle')
plt.plot(t,statesTotal[:,2],'r',linewidth=1,label='Car yaw angle')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('psi_ref-position [rad]',fontsize=15)
plt.grid(True)
plt.legend(loc='lower right',fontsize='small')

plt.subplot(3,1,2)
plt.plot(t,X_ref,'--b',linewidth=2,label='X_ref position')
plt.plot(t,statesTotal[:,4],'r',linewidth=1,label='Car X position')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('X-position [m]',fontsize=15)
plt.grid(True)
plt.legend(loc='lower right',fontsize='small')

plt.subplot(3,1,3)
plt.plot(t,Y_ref,'--b',linewidth=2,label='Y_ref position')
plt.plot(t,statesTotal[:,5],'r',linewidth=1,label='Car Y position')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('Y-position [m]',fontsize=15)
plt.grid(True)
plt.legend(loc='lower right',fontsize='small')
plt.show()


plt.subplot(3,1,1)
plt.plot(t,x_dot_ref,'--b',linewidth=2,label='x_dot_ref')
plt.plot(t,statesTotal[:,0],'r',linewidth=1,label='x_dot')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('x_dot [m/s]',fontsize=15)
plt.grid(True)
plt.legend(loc='center right',fontsize='small')

plt.subplot(3,1,2)
plt.plot(t,y_dot_ref,'--b',linewidth=2,label='y_dot_ref')
plt.plot(t,statesTotal[:,1],'r',linewidth=1,label='y_dot')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('y_dot [m/s]',fontsize=15)
plt.grid(True)
plt.legend(loc='center right',fontsize='small')

plt.subplot(3,1,3)
plt.plot(t,statesTotal[:,3],'r',linewidth=1,label='psi_dot')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('psi_dot [rad/s]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')
plt.show()


########################### Accelerations ######################################
plt.subplot(3,1,1)
plt.plot(t,accelerations_total[:,0],'b',linewidth=1,label='x_dot_dot')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('x_dot_dot [m/s^2]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

plt.subplot(3,1,2)
plt.plot(t,accelerations_total[:,1],'b',linewidth=1,label='y_dot_dot')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('y_dot_dot [m/s^2]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')

plt.subplot(3,1,3)
plt.plot(t,accelerations_total[:,2],'b',linewidth=1,label='psi_dot_dot')
plt.xlabel('t-time [s]',fontsize=15)
plt.ylabel('psi_dot_dot [rad/s^2]',fontsize=15)
plt.grid(True)
plt.legend(loc='upper right',fontsize='small')
plt.show()

exit()











##########################

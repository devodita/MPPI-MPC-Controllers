'''
LICENSE AGREEMENT

In relation to this Python file:

1. Copyright of this Python file is owned by the author: Mark Misin
2. This Python code can be freely used and distributed
3. The copyright label in this Python file such as

copyright=ax_main.text(x,y,'© Mark Misin Engineering',size=z)
that indicate that the Copyright is owned by Mark Misin MUST NOT be removed.

WARRANTY DISCLAIMER!

This Python file comes with absolutely NO WARRANTY! In no event can the author
of this Python file be held responsible for whatever happens in relation to this Python file.
For example, if there is a bug in the code and because of that a project, invention,
or anything else it was used for fails - the author is NOT RESPONSIBLE!
'''

import numpy as np
import matplotlib.pyplot as plt

class SupportFilesCar:
    ''' The following functions interact with the main file'''

    def __init__(self):
        ''' Load the constants that do not change'''

        # Constants
        g=9.81
        m=1500
        Iz=3000
        Cf=38000
        Cr=66000
        lf=2
        lr=3
        Ts=0.02
        mju=0.02 # friction coefficient

        ####################### Lateral control #################################

        outputs=4 # number of outputs
        inputs=2 # number of inputs
        hz = 10 # horizon period

        trajectory=3 # Choose 1, 2 or 3, nothing else
        version=2 # This is only for trajectory 3 (Choose 1 or 2)

        # Matrix weights for the cost function (They must be diagonal)

        if trajectory==3 and version==2:
            # Weights for trajectory 3, version 2
            Q=np.matrix('100 0 0 0;0 20000 0 0;0 0 1000 0;0 0 0 1000') # weights for outputs (all samples, except the last one)
            S=np.matrix('100 0 0 0;0 20000 0 0;0 0 1000 0;0 0 0 1000') # weights for the final horizon period outputs
            R=np.matrix('100 0;0 1') # weights for inputs
        elif trajectory==3:
            # Weights for trajectory 3, version 1
            Q=np.matrix('100 0 0 0;0 20000 0 0;0 0 1000 0;0 0 0 1000') # weights for outputs (all samples, except the last one)
            S=np.matrix('100 0 0 0;0 20000 0 0;0 0 1000 0;0 0 0 1000') # weights for the final horizon period outputs
            R=np.matrix('100 0;0 1') # weights for inputs
        else:
            # Weights for trajectories 1 & 2
            Q=np.matrix('1 0 0 0;0 200 0 0;0 0 50 0;0 0 0 50') # weights for outputs (all samples, except the last one)
            S=np.matrix('1 0 0 0;0 200 0 0;0 0 50 0;0 0 0 50') # weights for the final horizon period outputs
            R=np.matrix('100 0;0 1') # weights for inputs

        # Please do not modify the time_length!
        delay=0
        if trajectory==1:
            time_length = 60.
            x_lim=1000
            y_lim=1000
        elif trajectory ==2:
            time_length = 140.
            x_lim=1000
            y_lim=1000
        elif trajectory == 3:
            if version==1:
                x_lim=170
                y_lim=160
            else:
                x_lim=170*version
                y_lim=160*version
            first_section=14
            other_sections=14
            time_length=first_section+other_sections*10
            delay=np.zeros(12)
            for dly in range(1,len(delay)):
                delay[dly]=first_section+(dly-1)*other_sections
            # print(delay)
            # exit()
        else:
            print("trajectory: 1,2 or 3; version: 1 or 2")

        self.constants={'g':g,'m':m,'Iz':Iz,'Cf':Cf,'Cr':Cr,'lf':lf,'lr':lr,\
        'Ts':Ts,'mju':mju,'Q':Q,'S':S,'R':R,'outputs':outputs,'inputs':inputs,\
        'hz':hz,'delay':delay,'time_length':time_length,'trajectory':trajectory,\
        'version':version,'x_lim':x_lim,'y_lim':y_lim}
        # exit()
        return None

    def trajectory_generator(self,t):
        '''This method creates the trajectory for a car to follow'''

        Ts=self.constants['Ts']
        trajectory=self.constants['trajectory']
        x_lim=self.constants['x_lim']
        y_lim=self.constants['y_lim']

        # Define trajectories
        if trajectory==1:
            X=15*t
            Y=750/900**2*X**2+250

            # # Plot the world
            # plt.plot(X,Y,'b',linewidth=2,label='The trajectory')
            # plt.xlabel('X-position [m]',fontsize=15)
            # plt.ylabel('Y-position [m]',fontsize=15)
            # plt.grid(True)
            # plt.legend(loc='upper right',fontsize='small')
            # plt.xlim(0,x_lim)
            # plt.ylim(0,y_lim)
            # plt.xticks(np.arange(0,x_lim+1,int(x_lim/10)))
            # plt.yticks(np.arange(0,y_lim+1,int(y_lim/10)))
            # plt.show()
            #
            # plt.plot(t,X,'b',linewidth=2,label='X ref')
            # plt.plot(t,Y,'r',linewidth=2,label='Y ref')
            # plt.xlabel('t-position [s]',fontsize=15)
            # plt.ylabel('X,Y-position [m]',fontsize=15)
            # plt.grid(True)
            # plt.legend(loc='upper right',fontsize='small')
            # plt.xlim(0,t[-1])
            # plt.show()
            # # exit()

        elif trajectory==2:

            X1=15*t[0:int(40/Ts+1)]
            Y1=50*np.sin(2*np.pi*0.75/40*t[0:int(40/Ts+1)])+250

            X2=300*np.cos(2*np.pi*0.5/60*(t[int(40/Ts+1):int(100/Ts+1)]-40)-np.pi/2)+600
            Y2=300*np.sin(2*np.pi*0.5/60*(t[int(40/Ts+1):int(100/Ts+1)]-40)-np.pi/2)+500

            X3=600-15*(t[int(100/Ts+1):int(140/Ts+1)]-100)
            Y3=50*np.cos(2*np.pi*0.75/40*(t[int(100/Ts+1):int(140/Ts+1)]-100))+750

            X=np.concatenate((X1,X2),axis=0)
            Y=np.concatenate((Y1,Y2),axis=0)

            X=np.concatenate((X,X3),axis=0)
            Y=np.concatenate((Y,Y3),axis=0)

            # # Plot the world
            # plt.plot(X,Y,'b',linewidth=2,label='The trajectory')
            # plt.xlabel('X-position [m]',fontsize=15)
            # plt.ylabel('Y-position [m]',fontsize=15)
            # plt.xlim(0,x_lim)
            # plt.ylim(0,y_lim)
            # plt.xticks(np.arange(0,x_lim+1,int(x_lim/10)))
            # plt.yticks(np.arange(0,y_lim+1,int(y_lim/10)))
            # plt.grid(True)
            # plt.legend(loc='upper right',fontsize='small')
            # plt.show()
            #
            # plt.plot(t,X,'b',linewidth=2,label='X ref')
            # plt.plot(t,Y,'r',linewidth=2,label='Y ref')
            # plt.xlabel('t-position [s]',fontsize=15)
            # plt.ylabel('X,Y-position [m]',fontsize=15)
            # plt.grid(True)
            # plt.legend(loc='upper right',fontsize='small')
            # plt.xlim(0,t[-1])
            # plt.show()
            # # exit()

        else:

            # Attention! Do not let x_dot become 0 m/s. It is in the denominator.
            delay=self.constants['delay']
            version=self.constants['version']

            # X & Y levels
            f_x=np.array([0,60,110,140,160,110,40,10,40,70,110,150])*version
            f_y=np.array([40,20,20,60,100,140,140,80,60,60,90,90])*version

            # X & Y derivatives
            f_x_dot=np.array([2,1,1,1,0,-1,-1,0,1,1,1,1])*3*version
            f_y_dot=np.array([0,0,0,1,1,0,0,-1,0,0,0,0])*3*version


            X=[]
            Y=[]
            for i in range(0,len(delay)-1):
                # Extract the time elements for each section separately
                if i != len(delay)-2:
                    t_temp=t[int(delay[i]/Ts):int(delay[i+1]/Ts)]
                else:
                    t_temp=t[int(delay[i]/Ts):int(delay[i+1]/Ts+1)]

                # Generate data for a subtrajectory
                M=np.array([[1,t_temp[0],t_temp[0]**2,t_temp[0]**3],\
                            [1,t_temp[-1],t_temp[-1]**2,t_temp[-1]**3],\
                            [0,1,2*t_temp[0],3*t_temp[0]**2],\
                            [0,1,2*t_temp[-1],3*t_temp[-1]**2]])

                c_x=np.array([[f_x[i]],[f_x[i+1]-f_x_dot[i+1]*Ts],[f_x_dot[i]],[f_x_dot[i+1]]])
                c_y=np.array([[f_y[i]],[f_y[i+1]-f_y_dot[i+1]*Ts],[f_y_dot[i]],[f_y_dot[i+1]]])


                a_x=np.matmul(np.linalg.inv(M),c_x)
                a_y=np.matmul(np.linalg.inv(M),c_y)

                # Compute X and Y values
                X_temp=a_x[0][0]+a_x[1][0]*t_temp+a_x[2][0]*t_temp**2+a_x[3][0]*t_temp**3
                Y_temp=a_y[0][0]+a_y[1][0]*t_temp+a_y[2][0]*t_temp**2+a_y[3][0]*t_temp**3

                # Concatenate X and Y values
                X=np.concatenate([X,X_temp])
                Y=np.concatenate([Y,Y_temp])

            # Round the numbers to avoid numerical errors
            X=np.round(X,8)
            Y=np.round(Y,8)

            # # Plot the world
            # plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95,wspace=0.15,hspace=0.2)
            # plt.plot(X,Y,'b',linewidth=2,label='The ref trajectory')
            # plt.xlabel('X-position [m]',fontsize=15)
            # plt.ylabel('Y-position [m]',fontsize=15)
            # plt.grid(True)
            # plt.legend(loc='upper right',fontsize='small')
            # plt.xlim(0,x_lim)
            # plt.ylim(0,y_lim)
            # plt.xticks(np.arange(0,x_lim+1,int(x_lim/10)))
            # plt.yticks(np.arange(0,y_lim+1,int(y_lim/10)))
            # plt.show()
            #
            # plt.subplots_adjust(left=0.05,bottom=0.05,right=0.95,top=0.95,wspace=0.15,hspace=0.2)
            # plt.plot(t,X,'b',linewidth=2,label='X ref')
            # plt.plot(t,Y,'r',linewidth=2,label='Y ref')
            # plt.xlabel('t-position [s]',fontsize=15)
            # plt.ylabel('X,Y-position [m]',fontsize=15)
            # plt.grid(True)
            # plt.legend(loc='upper right',fontsize='small')
            # plt.xlim(0,t[-1])
            # plt.show()
            # # exit()

        # Vector of x and y changes per sample time
        dX=X[1:len(X)]-X[0:len(X)-1]
        dY=Y[1:len(Y)]-Y[0:len(Y)-1]

        X_dot=dX/Ts
        Y_dot=dY/Ts
        X_dot=np.concatenate(([X_dot[0]],X_dot),axis=0)
        Y_dot=np.concatenate(([Y_dot[0]],Y_dot),axis=0)

        # Define the reference yaw angles
        psi=np.zeros(len(X))
        psiInt=psi
        psi[0]=np.arctan2(dY[0],dX[0])
        psi[1:len(psi)]=np.arctan2(dY[0:len(dY)],dX[0:len(dX)])

        # We want the yaw angle to keep track the amount of rotations
        dpsi=psi[1:len(psi)]-psi[0:len(psi)-1]
        psiInt[0]=psi[0]
        for i in range(1,len(psiInt)):
            if dpsi[i-1]<-np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]+2*np.pi)
            elif dpsi[i-1]>np.pi:
                psiInt[i]=psiInt[i-1]+(dpsi[i-1]-2*np.pi)
            else:
                psiInt[i]=psiInt[i-1]+dpsi[i-1]


        x_dot_body=np.cos(psiInt)*X_dot+np.sin(psiInt)*Y_dot
        y_dot_body=-np.sin(psiInt)*X_dot+np.cos(psiInt)*Y_dot
        y_dot_body=np.round(y_dot_body)

        # # Plot the body frame velocity
        # # plt.plot(t,x_dot_body,'g',linewidth=2,label='x_dot ref')
        # plt.plot(t,X_dot,'b',linewidth=2,label='X_dot ref')
        # plt.plot(t,Y_dot,'r',linewidth=2,label='Y_dot ref')
        # plt.xlabel('t [s]',fontsize=15)
        # plt.ylabel('X_dot_ref, Y_dot_ref [m/s]',fontsize=15)
        # plt.grid(True)
        # plt.legend(loc='upper right',fontsize='small')
        # plt.show()
        #
        # # Plot the reference yaw angle
        # plt.plot(t,psiInt,'g',linewidth=2,label='Psi ref')
        # plt.xlabel('t [s]',fontsize=15)
        # plt.ylabel('Psi_ref [rad]',fontsize=15)
        # plt.grid(True)
        # plt.legend(loc='upper right',fontsize='small')
        # plt.show()
        # exit()

        return x_dot_body,y_dot_body,psiInt,X,Y

    def state_space(self,states,delta,a):
        '''This function forms the state space matrices and transforms them in the discrete form'''

        # Get the necessary constants
        g=self.constants['g']
        m=self.constants['m']
        Iz=self.constants['Iz']
        Cf=self.constants['Cf']
        Cr=self.constants['Cr']
        lf=self.constants['lf']
        lr=self.constants['lr']
        Ts=self.constants['Ts']
        mju=self.constants['mju']

        # Get the necessary states
        x_dot=states[0]
        y_dot=states[1]
        psi=states[2]

        # Get the state space matrices for the control
        A11=-mju*g/x_dot
        A12=Cf*np.sin(delta)/(m*x_dot)
        A14=Cf*lf*np.sin(delta)/(m*x_dot)+y_dot
        A22=-(Cr+Cf*np.cos(delta))/(m*x_dot)
        A24=-(Cf*lf*np.cos(delta)-Cr*lr)/(m*x_dot)-x_dot
        A34=1
        A42=-(Cf*lf*np.cos(delta)-lr*Cr)/(Iz*x_dot)
        A44=-(Cf*lf**2*np.cos(delta)+lr**2*Cr)/(Iz*x_dot)
        A51=np.cos(psi)
        A52=-np.sin(psi)
        A61=np.sin(psi)
        A62=np.cos(psi)

        B11=-1/m*np.sin(delta)*Cf
        B12=1
        B21=1/m*np.cos(delta)*Cf
        B41=1/Iz*np.cos(delta)*Cf*lf


        A=np.array([[A11, A12, 0, A14, 0, 0],[0, A22, 0, A24, 0, 0],[0, 0, 0, A34, 0, 0],\
        [0, A42, 0, A44, 0, 0],[A51, A52, 0, 0, 0, 0],[A61, A62, 0, 0, 0, 0]])
        B=np.array([[B11, B12],[B21, 0],[0, 0],[B41, 0],[0, 0],[0, 0]])
        C=np.array([[1, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0],[0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 1]])
        D=np.array([[0, 0],[0, 0],[0, 0],[0, 0]])

        # Discretise the system (forward Euler)
        Ad=np.identity(np.size(A,1))+Ts*A
        Bd=Ts*B
        Cd=C
        Dd=D

        return Ad, Bd, Cd, Dd

    def augmented_matrices(self, Ad, Bd, Cd, Dd):

        A_aug=np.concatenate((Ad,Bd),axis=1)
        temp1=np.zeros((np.size(Bd,1),np.size(Ad,1)))
        temp2=np.identity(np.size(Bd,1))
        temp=np.concatenate((temp1,temp2),axis=1)

        A_aug=np.concatenate((A_aug,temp),axis=0)
        B_aug=np.concatenate((Bd,np.identity(np.size(Bd,1))),axis=0)
        C_aug=np.concatenate((Cd,np.zeros((np.size(Cd,0),np.size(Bd,1)))),axis=1)
        D_aug=Dd

        return A_aug, B_aug, C_aug, D_aug

    def mpc_simplification(self, Ad, Bd, Cd, Dd, hz, x_aug_t, du):
        '''This function creates the compact matrices for Model Predictive Control'''
        # db - double bar
        # dbt - double bar transpose
        # dc - double circumflex

        A_aug, B_aug, C_aug, D_aug=self.augmented_matrices(Ad, Bd, Cd, Dd)

        Q=self.constants['Q']
        S=self.constants['S']
        R=self.constants['R']
        Cf=self.constants['Cf']
        g=self.constants['g']
        m=self.constants['m']
        mju=self.constants['mju']
        lf=self.constants['lf']
        inputs=self.constants['inputs']

        ############################### Constraints #############################
        d_delta_max=np.pi/300
        d_a_max=0.1
        d_delta_min=-np.pi/300
        d_a_min=-0.1

        ub_global=np.zeros(inputs*hz)
        lb_global=np.zeros(inputs*hz)

        # Only works for 2 inputs
        for i in range(0,inputs*hz):
            if i%2==0:
                ub_global[i]=d_delta_max
                lb_global[i]=-d_delta_min
            else:
                ub_global[i]=d_a_max
                lb_global[i]=-d_a_min

        ub_global=ub_global[0:inputs*hz]
        lb_global=lb_global[0:inputs*hz]
        ublb_global=np.concatenate((ub_global,lb_global),axis=0)

        I_global=np.eye(inputs*hz)
        I_global_negative=-I_global
        I_mega_global=np.concatenate((I_global,I_global_negative),axis=0)

        y_asterisk_max_global=[]
        y_asterisk_min_global=[]

        C_asterisk=np.matrix('1 0 0 0 0 0 0 0;\
                        0 1 0 0 0 0 0 0;\
                        0 0 0 0 0 0 1 0;\
                        0 0 0 0 0 0 0 1')

        C_asterisk_global=np.zeros((np.size(C_asterisk,0)*hz,np.size(C_asterisk,1)*hz))

        #########################################################################

        CQC=np.matmul(np.transpose(C_aug),Q)
        CQC=np.matmul(CQC,C_aug)

        CSC=np.matmul(np.transpose(C_aug),S)
        CSC=np.matmul(CSC,C_aug)

        QC=np.matmul(Q,C_aug)
        SC=np.matmul(S,C_aug)

        Qdb=np.zeros((np.size(CQC,0)*hz,np.size(CQC,1)*hz))
        Tdb=np.zeros((np.size(QC,0)*hz,np.size(QC,1)*hz))
        Rdb=np.zeros((np.size(R,0)*hz,np.size(R,1)*hz))
        Cdb=np.zeros((np.size(B_aug,0)*hz,np.size(B_aug,1)*hz))
        Adc=np.zeros((np.size(A_aug,0)*hz,np.size(A_aug,1)))

        ######################### Advanced LPV ##################################
        A_product=A_aug
        states_predicted_aug=x_aug_t
        A_aug_collection=np.zeros((hz,np.size(A_aug,0),np.size(A_aug,1)))
        B_aug_collection=np.zeros((hz,np.size(B_aug,0),np.size(B_aug,1)))
        #########################################################################

        for i in range(0,hz):
            if i == hz-1:
                Qdb[np.size(CSC,0)*i:np.size(CSC,0)*i+CSC.shape[0],np.size(CSC,1)*i:np.size(CSC,1)*i+CSC.shape[1]]=CSC
                Tdb[np.size(SC,0)*i:np.size(SC,0)*i+SC.shape[0],np.size(SC,1)*i:np.size(SC,1)*i+SC.shape[1]]=SC
            else:
                Qdb[np.size(CQC,0)*i:np.size(CQC,0)*i+CQC.shape[0],np.size(CQC,1)*i:np.size(CQC,1)*i+CQC.shape[1]]=CQC
                Tdb[np.size(QC,0)*i:np.size(QC,0)*i+QC.shape[0],np.size(QC,1)*i:np.size(QC,1)*i+QC.shape[1]]=QC

            Rdb[np.size(R,0)*i:np.size(R,0)*i+R.shape[0],np.size(R,1)*i:np.size(R,1)*i+R.shape[1]]=R

            ########################### Advanced LPV ############################
            Adc[np.size(A_aug,0)*i:np.size(A_aug,0)*i+A_aug.shape[0],0:0+A_aug.shape[1]]=A_product
            A_aug_collection[i][:][:]=A_aug
            B_aug_collection[i][:][:]=B_aug
            #####################################################################

            ######################## Constraints ################################
            x_dot_max=30
            if 0.17*states_predicted_aug[0][0] < 3:
                y_dot_max=0.17*states_predicted_aug[0][0]
            else:
                y_dot_max=3
            delta_max=np.pi/6
            Fyf=Cf*(states_predicted_aug[6][0]-states_predicted_aug[1][0]/states_predicted_aug[0][0]-lf*states_predicted_aug[3][0]/states_predicted_aug[0][0])
            a_max=1+(Fyf*np.sin(states_predicted_aug[6][0])+mju*m*g)/m-states_predicted_aug[3][0]*states_predicted_aug[1][0]
            x_dot_min=1
            if -0.17*states_predicted_aug[0][0] > -3:
                y_dot_min=-0.17*states_predicted_aug[0][0]
            else:
                y_dot_min=-3
            delta_min=-np.pi/6
            a_min=-4+(Fyf*np.sin(states_predicted_aug[6][0])+mju*m*g)/m-states_predicted_aug[3][0]*states_predicted_aug[1][0]

            y_asterisk_max=np.array([x_dot_max,y_dot_max,delta_max,a_max])
            y_asterisk_min=np.array([x_dot_min,y_dot_min,delta_min,a_min])

            y_asterisk_max_global=np.concatenate((y_asterisk_max_global,y_asterisk_max),axis=0)
            y_asterisk_min_global=np.concatenate((y_asterisk_min_global,y_asterisk_min),axis=0)

            C_asterisk_global[np.size(C_asterisk,0)*i:np.size(C_asterisk,0)*i+C_asterisk.shape[0],np.size(C_asterisk,1)*i:np.size(C_asterisk,1)*i+C_asterisk.shape[1]]=C_asterisk


            #####################################################################

            ######################### Advanced LPV ##############################
            if i<hz-1:
                du1=du[inputs*(i+1)][0]
                du2=du[inputs*(i+1)+inputs-1][0]
                states_predicted_aug=np.matmul(A_aug,states_predicted_aug)+np.matmul(B_aug,np.transpose([[du1,du2]]))
                states_predicted=np.transpose(states_predicted_aug[0:6])[0]
                delta_predicted=states_predicted_aug[6][0]
                a_predicted=states_predicted_aug[7][0]
                Ad, Bd, Cd, Dd=self.state_space(states_predicted,delta_predicted,a_predicted)
                A_aug, B_aug, C_aug, D_aug=self.augmented_matrices(Ad, Bd, Cd, Dd)
                A_product=np.matmul(A_aug,A_product)

        for i in range(0,hz):
            for j in range(0,hz):
                if j<=i:
                    AB_product=np.eye(np.shape(A_aug)[0])
                    for ii in range(i,j-1,-1):
                        if ii>j:
                            AB_product=np.matmul(AB_product,A_aug_collection[ii][:][:])
                        else:
                            AB_product=np.matmul(AB_product,B_aug_collection[ii][:][:])
                    Cdb[np.size(B_aug,0)*i:np.size(B_aug,0)*i+B_aug.shape[0],np.size(B_aug,1)*j:np.size(B_aug,1)*j+B_aug.shape[1]]=AB_product

        #########################################################################

        ####################### Constraints #####################################

        Cdb_constraints=np.matmul(C_asterisk_global,Cdb)
        Cdb_constraints_negative=-Cdb_constraints
        Cdb_constraints_global=np.concatenate((Cdb_constraints,Cdb_constraints_negative),axis=0)

        Adc_constraints=np.matmul(C_asterisk_global,Adc)
        Adc_constraints_x0=np.transpose(np.matmul(Adc_constraints,x_aug_t))[0]
        y_max_Adc_difference=y_asterisk_max_global-Adc_constraints_x0
        y_min_Adc_difference=-y_asterisk_min_global+Adc_constraints_x0
        y_Adc_difference_global=np.concatenate((y_max_Adc_difference,y_min_Adc_difference),axis=0)

        G=np.concatenate((I_mega_global,Cdb_constraints_global),axis=0)
        ht=np.concatenate((ublb_global,y_Adc_difference_global),axis=0)

        #######################################################################

        Hdb=np.matmul(np.transpose(Cdb),Qdb)
        Hdb=np.matmul(Hdb,Cdb)+Rdb

        temp=np.matmul(np.transpose(Adc),Qdb)
        temp=np.matmul(temp,Cdb)

        temp2=np.matmul(-Tdb,Cdb)
        Fdbt=np.concatenate((temp,temp2),axis=0)

        return Hdb,Fdbt,Cdb,Adc,G,ht

    def open_loop_new_states(self,states,delta,a):
        '''This function computes the new state vector for one sample time later'''

        # Get the necessary constants
        g=self.constants['g']
        m=self.constants['m']
        Iz=self.constants['Iz']
        Cf=self.constants['Cf']
        Cr=self.constants['Cr']
        lf=self.constants['lf']
        lr=self.constants['lr']
        Ts=self.constants['Ts']
        mju=self.constants['mju']

        current_states=states
        new_states=current_states
        x_dot=current_states[0]
        y_dot=current_states[1]
        psi=current_states[2]
        psi_dot=current_states[3]
        X=current_states[4]
        Y=current_states[5]

        sub_loop=30  #Chops Ts into 30 pieces
        for i in range(0,sub_loop):

            # Compute lateral forces
            Fyf=Cf*(delta-y_dot/x_dot-lf*psi_dot/x_dot)
            Fyr=Cr*(-y_dot/x_dot+lr*psi_dot/x_dot)

            # Compute the the derivatives of the states
            x_dot_dot=a+(-Fyf*np.sin(delta)-mju*m*g)/m+psi_dot*y_dot
            y_dot_dot=(Fyf*np.cos(delta)+Fyr)/m-psi_dot*x_dot
            psi_dot=psi_dot
            psi_dot_dot=(Fyf*lf*np.cos(delta)-Fyr*lr)/Iz
            X_dot=x_dot*np.cos(psi)-y_dot*np.sin(psi)
            Y_dot=x_dot*np.sin(psi)+y_dot*np.cos(psi)

            # Update the state values with new state derivatives
            x_dot=x_dot+x_dot_dot*Ts/sub_loop
            y_dot=y_dot+y_dot_dot*Ts/sub_loop
            psi=psi+psi_dot*Ts/sub_loop
            psi_dot=psi_dot+psi_dot_dot*Ts/sub_loop
            X=X+X_dot*Ts/sub_loop
            Y=Y+Y_dot*Ts/sub_loop

        # Take the last states
        new_states[0]=x_dot
        new_states[1]=y_dot
        new_states[2]=psi
        new_states[3]=psi_dot
        new_states[4]=X
        new_states[5]=Y

        return new_states,x_dot_dot,y_dot_dot,psi_dot_dot

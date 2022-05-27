"""
Created on : 21 feb 2022

@author: Sumit Kumar Singh
#For more information and citations :   Singh, S.K.; Haginaka, H.; Jackin, B.J.; Kinashi, K.; Tsutsumi, N.; Sakai, W. 
                                        Generation of Ince–Gaussian Beams Using Azocarbazole Polymer CGH.
                                        J. Imaging 2022, 8, 144.  https://doi.org/10.3390/jimaging8050144
                                        
"""

#Import Required Packages
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv2
import scipy.misc
from numpy import linalg as la
from scipy.special import gamma

#Define cartesian meshgrid and Gaussian beam parameters

L=0.01                                       #size of meshgrid
N=512                                        #dimension of meshgrid

x,y=np.meshgrid(np.linspace(-L,L,N),np.linspace(-L,L,N))
rr=np.sqrt(x**2 +y**2)
w0=0.003                                     #Beam waist (3mm)
k = 2*np.pi/532.0e-9;                        # Wavenumber of light (Wavelength =532 nm)
z0 = k*w0**2.0/2;                            # Calculate the Rayleigh range
z=0                                          #Propagation Distance
w = w0 * np.sqrt(1.0 + z**2/z0**2);


#calculate elliptical coordinate system
a=1
b=0.9999955  
#print(b)           
f0=np.sqrt((a**2-b**2))             #manually fix (f0=sqrt(a**2 -b**2)=3mm for IG beam)
#print((2*f0**2)/w0**2,"elliptical parameter")
#b=0                                        #(f0=1)for Hermite Gaussian Beam
#b=0.999999999                              #(f0 =0) for Laguerre gaussian beam

####### conversion of cartesian coordinate to elliptical coordinate ###########

c2=(a**2-b**2)*(w0/w)**2
c=2*c2;
x2=x**2;
y2=y**2;
B=x2+y2-c2
del2= B**2 + 2*c*y2
del1=np.sqrt(del2)
p=(-B+del1)/c;
p=np.where(p>1,1,p)
p=np.sqrt(p);
et0=np.arcsin(p);
eta=x;
q=(-B-del1)/c;
del2=q**2-q; 
del1=np.sqrt(del2);
zeta=(np.log(1-2*q+2*del1))/2

for i in range(N):
    for j in range(N):
        if ((x[i,j]>=0).any() and (y[i,j]>=0).any()):
            eta[i,j]=et0[i,j]
        elif ((x[i,j]<0).any() and (y[i,j]>=0).any()):
            eta[i][j]=np.pi-et0[i][j]
        elif ((x[i,j]<=0).any() and (y[i,j]<0).any()):
            eta[i][j]=np.pi+et0[i][j]
        elif ((x[i,j]>0).any() and (y[i,j]<0).any()):
            eta[i][j]=2*np.pi-et0[i][j]


####### generation of Ince-gaussian beam #######

def incegauss(p,m,beam):
    
    q=(2* f0**2)/(w0**2)
    
    z1=eta
    c1,c2=np.shape((z1));
    z1=np.transpose(z1[:])

    z2=1j*zeta
    c1,c2=np.shape((z2));
    z2=np.transpose(z2[:])
    
    if ((-1)**(m-p) != 1):
        print(' ERROR!!! p and m does not have same parity')
    if (m<1 and m>p):
        print('ERROR! ERROR! wrong range of m')
        
    #for even Ince-gaussian Beam
    
    if (beam==2):
        #for even indices
        if (p%2 ==0):
            j=p/2
            N=int(j+1)
            n=int(m/2 +1)
            m1=[]
            m2=[]
            m2.insert(0,2*q*j)
            m3=[]
            m3.insert(0,0)
            for i in range(1,N,1):    
                m1.append(q*(j+i))
            m1=np.diag(m1,1)
            for i in range(1,N-1):    
                m2.append(q*(j-i))
            m2=np.diag(m2,-1)
            for i in range(0,N-1):    
                m3.append(4*(i+1)**2)
            m3=np.diag(m3)
            M=m1+m2+m3               #matrix is generated for finding the coefficients of recurrence relation
                         
            ets,A=(la.eig(M))        #Calculating eigenfunctions
                        
            index=np.argsort(ets)
            ets=np.sort(ets)
            ets=ets.reshape(N,1)
            A=A[:,index]
            
            #Normalization for ince polynomial
            
            mv=np.arange(2,p+1,2)
            N22=[]
            for i in range(1,int(p/2 +1)):
                N22.append(A[ i ,n-1])
            
            N2=np.sqrt(A[0,n-1]**2*2*gamma(p/2+1)**2+ np.sum((np.sqrt(gamma((p+mv)/2+1)*gamma((p-mv)/2+1) )*N22)**2))
            NS=np.sign(np.sum(A[:,n-1]));
            A=A/N2*NS;
            r=np.arange(0,N,1)
                        
            R,X=np.meshgrid(r,z1)
            
            #defining Ince-Polynomial
            IP1=np.dot(np.cos(2*X*R) , ((A[:,n-1].reshape(N,1))))
            
            IP1=np.transpose(IP1.reshape(c1,c2))
            dIP=np.dot(-2*R*np.sin(2*X*R) , ((A[:,n-1].reshape(N,1))))
                        
            etha=ets[n-1]
            coef=A[:,n-1]
            
            R1,X1=np.meshgrid(r,z2)
            
            IP2=(np.dot(np.cos(2*X1*R1) , ((A[:,n-1].reshape(N,1)))))
            
            IP2=np.transpose(IP2.reshape(c1,c2))
            
            R2,X2=np.meshgrid(r,0)
            
            IP3=(np.dot(np.cos(2*X2*R2) , ((A[:,n-1].reshape(N,1)))))
            
            R4,X4=np.meshgrid(r,np.pi/2)
            
            IP4=(np.dot( np.cos(2*X4*R4) , ((A[:,n-1].reshape(N,1)))  ))
            #Normalization for Ince-Gaussian beam
            Norm = (-1)**(m/2)*np.sqrt(2)*gamma(p/2+1)*coef[0] *np.sqrt(2/np.pi)/w0/IP3/IP4
            
            
        else:
            #for odd indices
            j=(p-1)/2
            N=int(j+1)
            n=int((m +1)/2)
            m1=[]
            m2=[]
            m3=[]
            m3.insert(0,(q/2 +(p*(q/2))+1))
        
            for i in range(0,N-1,1):    
                m1.append((q/2)*(p+(2*i +3)))
            m1=np.diag(m1,1)
            for i in range(1,N):    
                m2.append((q/2)*(p-(2*i -1)))
            m2=np.diag(m2,-1)
            for i in range(1,N):    
                m3.append((2*i+1)**2)
            m3=np.diag(m3)
            M=m1+m2+m3
                
            ets,A=(la.eig(M))
        
            index=np.argsort(ets)
            
            ets=np.sort(ets)
            
            ets=ets.reshape(N,1)
            
            A=A[:,index]
            
            mv=np.arange(1,p+1,2)
            N2=np.sqrt(np.sum( ( np.sqrt(gamma((p+mv)/2+1)*gamma((p-mv)/2+1) )*A[:,n-1])**2 ))
            NS=np.sign(np.sum(A[:,n-1]));
            A=A/N2*NS;
            
            r=np.arange(1,2*N,2)
            
            R,X=np.meshgrid(r,z1)
        
            IP1=np.dot(np.cos(X*R) , ((A[:,n-1].reshape(N,1))))
        
            IP1=np.transpose(IP1.reshape(c1,c2))
            dIP=np.dot(-R*np.sin(X*R) , ((A[:,n-1].reshape(N,1))))
            
            etha=ets[n-1]
            coef=A[:,n-1]
        
            R1,X1=np.meshgrid(r,z2)
        
            
            IP2=(np.dot(np.cos(X1*R1) , ((A[:,n-1].reshape(N,1)))))
        
            IP2=np.transpose(IP2.reshape(c1,c2))
            
            R2,X2=np.meshgrid(r,0)
            
            IP3=(np.dot(np.cos(X2*R2) , ((A[:,n-1].reshape(N,1)))))
            R4,X4=np.meshgrid(r,np.pi/2)
            
            IP4=(np.dot(-R4*np.sin(X4*R4) , ((A[:,n-1].reshape(N,1)))))
        
            Norm = (-1)**((m+1)/2) * gamma((p+1)/2+1) * np.sqrt(4*q/np.pi) * coef[0] / w0 / IP3 / IP4
           
    #for odd Ince-gaussian beam        
    else:
        #for even indices
        if (p%2 ==0):
            j=p/2
            N=int(j+1)
            n=int(m/2)
            m1=[]
            m2=[]
            m3=[]    
            for i in range(2,N,1):    
                m1.append(q*(j+i))
            m1=np.diag(m1,1)
            for i in range(1,N-1):    
                m2.append(q*(j-i))
            m2=np.diag(m2,-1)
            for i in range(0,N-1):    
                m3.append(4*(i+1)**2)
            m3=np.diag(m3)
            M=m1+m2+m3
                        
            ets,A=(la.eig(M))
                        
            index=np.argsort(ets)
            ets=np.sort(ets)
           
            ets=ets.reshape(N-1,1)
            A=A[:,index]
            r=np.arange(1,N,1)
            mv=np.arange(2,p+1,2)
            N2=np.sqrt(np.sum((np.sqrt(gamma((p+mv)/2+1)*gamma((p-mv)/2+1) )*A[:,n-1])**2 ))
            NS=np.sign(np.sum(r*A[:,n-1]))
            A=A/N2*NS;
                        
            R,X=np.meshgrid(r,z1)
           
            IP1=np.dot(np.sin(2*X*R) , ((A[:,n-1].reshape(N-1,1))))
            
            IP1=np.transpose(IP1.reshape(c1,c2))
                        
            etha=ets[n-1]
            coef=A[:,n-1]
            
            R1,X1=np.meshgrid(r,z2)
            
            IP2=(np.dot(np.sin(2*X1*R1) , ((A[:,n-1].reshape(N-1,1)))))
            
            IP2=np.transpose(IP2.reshape(c1,c2))
            
            R2,X2=np.meshgrid(r,0)
            
            IP3=(np.dot(2*R2*np.cos(2*X2*R2) , ((A[:,n-1].reshape(N-1,1)))))
            R4,X4=np.meshgrid(r,np.pi/2)
            
            IP4=(np.dot( 2*R4*np.cos(2*X4*R4) , ((A[:,n-1].reshape(N-1,1)))  ))
        
            Norm = (-1)**(m/2)*np.sqrt(2)*q*gamma((p+2)/2+1)*coef[0] *np.sqrt(2/np.pi)/w0/ IP3 / IP4
           
        #for odd indices    
        else:
            
            j=int((p-1)/2)
            N=int(j+1)
            n=int((m +1)/2)
            m1=[]
            m2=[]
            m3=[]
            m3.insert(0,(-q/2 -(p*(q/2))+1))
        
            for i in range(0,N-1,1):    
                m1.append((q/2)*(p+(2*i +3)))
            m1=np.diag(m1,1)
            for i in range(1,N):    
                m2.append((q/2)*(p-(2*i -1)))
            m2=np.diag(m2,-1)
            for i in range(1,N):    
                m3.append((2*i+1)**2)
            m3=np.diag(m3)
            M=m1+m2+m3
                    
            ets,A=(la.eig(M))
                    
            index=np.argsort(ets)
            ets=np.sort(ets)
            ets=ets.reshape(N,1)
            A=A[:,index]
            r=np.arange(1,2*N,2)
            mv=np.arange(1,p+1,2)
            N2=np.sqrt(np.sum( ( np.sqrt(gamma((p+mv)/2+1)*gamma((p-mv)/2+1) )*A[:,n-1])**2 ))
            NS=np.sign(np.sum(r*A[:,n-1]));
            A=A/N2*NS;
        
            R,X=np.meshgrid(r,z1)
        
            IP1=np.dot(np.sin(X*R) , ((A[:,n-1].reshape(N,1))))
        
            IP1=np.transpose(IP1.reshape(c1,c2))
        
            etha=ets[n-1]
            coef=A[:,n-1]
        
            R1,X1=np.meshgrid(r,z2)
        
            IP2=(np.dot(np.sin(X1*R1) , ((A[:,n-1].reshape(N,1)))))
        
            IP2=np.transpose(IP2.reshape(c1,c2))
                
            R2,X2=np.meshgrid(r,np.pi/2)
            
            IP3=(np.dot(np.sin(X2*R2) , ((A[:,n-1].reshape(N,1)))))
            R4,X4=np.meshgrid(r,0)
            
            IP4=(np.dot(R4*np.cos(X4*R4) , ((A[:,n-1].reshape(N,1)))))
            Norm = (-1)**((m-1)/2) * gamma((p+1)/2+1) * np.sqrt(4*q/np.pi) * coef[0] / w0 / IP3 / IP4  
            
    add=Norm*(IP1*IP2*np.exp(-(rr /w0)**2 ))*(w0*np.exp(1j*k*z) *np.exp((1j*k*z*rr**2)/((2*((z**2)+((z0)**2))))) *np.exp(-1j*(p+1)*np.arctan(z/z0)))/w        
    
    return add                       
    

#Helical Ince-Gaussian Beam
hel=(incegauss(12,8,2)+ (incegauss(12,8,1)))        #try:(12,8,2) (16,8,1)

#Add carrier frequency   
def car():
    x,y=np.meshgrid(np.linspace(-L,L,N),np.linspace(-L,L,N))
    carr=np.exp(1j*x*550/L)  #select carrier frequency 
    return carr

#manually enter mode number: beam=2(even ince-gaussian), beam=1(odd ince-gaussian)
beam=int(input('enter 2 for even Ince-Gaussian beam or Enter1 for odd Gauss ince beam== '))
print("enter mode number to generate IG beam")
p=int(input("p="))
m=int(input("m="))

    
#ploting of Intensity and phase of Ince-Gauss Beam
beam=incegauss(p,m,beam)
add=beam*car()          #(Replace beam by hel for helical Ince-Gaussian beam)
I1=abs(add)**2          #INTENSITY of Ince-gaussian beam
phs=np.angle(beam)      #Phase of Ince-Gaussian Beam
phs1=np.angle(add)      #carrier frequency added phase

#Normalization
I=(I1-np.min(I1))/(np.max(I1)-np.min(I1))
hol=(((phs1-np.min(phs1))/(np.max(phs1)-np.min(phs1))) * 255).astype('uint8')

#plotting 

plt.imshow(phs,cmap='gray') 
plt.colorbar()
plt.show()
plt.imshow(hol,cmap='gray')
plt.colorbar()
plt.show()
plt.imshow(I,cmap='hot') 
plt.colorbar()
plt.show()
print("Congratulations!! required  phase , Hologram and Intensity profile of IG beam have been  Successfully generated.")   

 
# =============================================================================
#     #Checking the orthogonality of ince gauss beam
#     
#     mod=[]
#     for i in range(3,9,2):
#         for j in range(3,i+2,2):
#             mod.append(incegauss(i,j,1))
#                 
#     ozm = np.real([[np.sum(mod[i]*np.conj(mod[j]))*(((2*L)/(N-1))**2) for j in range(len(mod))] for i in range(len(mod))])
#     
#     ortho=(incegauss(4,4,1)*np.conj(incegauss(4,4,1)))
#     
#     print((abs(np.sum(ortho)*((2*L)/(N-1))**2)))       #print orthogonality relation between two specific beams defined above in ortho
#     
#     #plt.imshow(np.log10(ozm))           #plotting orthogonality relation
#     #plt.colorbar()
# 
# =============================================================================

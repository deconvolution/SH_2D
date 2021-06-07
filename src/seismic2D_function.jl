module SH_solver
using MAT,Plots,Dates,TimerOutputs,WriteVTK,DataFrames,CSV,ProgressMeter

const USE_GPU=false  # Use GPU? If this is set false, then no GPU needs to be available
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2);
else
    @init_parallel_stencil(Threads, Float64, 2);
end

function meshgrid(x,y)
    x2=zeros(length(x),length(y));
    y2=x2;
    x2=repeat(x,1,length(y));
    y2=repeat(reshape(y,1,length(y)),length(x),1);
    return x2,y2
end

function write2mat(path,var)
    file=matopen(path,"w");
    write(file,"data",data);
    close(file);
    return nothing
end

function readmat(path,var)
    file=matopen(path);
    tt=read(file,var);
    close(file)
    return tt
end

function rickerWave(freq,dt,ns,M)
    ## calculate scale
    E=10 .^(5.24+1.44 .*M);
    s=sqrt(E.*freq/.299);

    t=dt:dt:dt*ns;
    t0=1 ./freq;
    t=t .-t0;
    ricker=s .*(1 .-2*pi^2*freq .^2*t .^2).*exp.(-pi^2*freq^2 .*t .^2);
    ricker=ricker;
    ricker=Float32.(ricker);
    return ricker
end
##
@parallel_indices (iz) function x_2_end(in,out)
out[:,iz]=in[2:end,iz];
return nothing
end

@parallel_indices (ix) function z_2_end(in,out)
out[ix,:]=in[ix,2:end];
return nothing
end

@timeit ti "mono_2D" function mono_2D_sh(input)
global data

d0=Dates.now();
# source number
ns=length(input.source.s3);

# create main folder
if isdir(input.visualization.path)==0
    mkdir(input.visualization.path);
end

# create folder for picture
n_picture=1;
path_pic=string(input.visualization.path,"/pic/");
if input.visualization.plot_interval!=0
    if isdir(path_pic)==0
        mkdir(path_pic);
    end
    # initialize pvd
    pvd=paraview_collection(string(input.visualization.path,"/time_info"));
end

# create folder for model
path_model=string(input.visualization.path,"/model/");
if isdir(path_model)==0
    mkdir(path_model)
end
vtkfile=vtk_grid(string(path_model,"material_parameter"),input.dims.X,input.dims.Z);
vtkfile["C44"]=input.material_parameter.C44;
vtkfile["ts2"]=input.material_parameter.ts2;
vtkfile["phi2"]=input.material_parameter.phi2;
vtk_save(vtkfile);
CSV.write(string(path_model,"/receiver location.csv"),
DataFrame([input.receiver.r1t input.receiver.r3t]));
CSV.write(string(path_model,"/source location.csv"),
DataFrame([input.source.s1t input.source.s3t]));

# create folder for rec
path_rec=string(input.visualization.path,"/rec/");
if input.receiver.r3!=nothing
    if isdir(path_rec)==0
        mkdir(path_rec)
    end
end

# PML
vmax=sqrt.((input.material_parameter.C44) ./input.material_parameter.rho);
beta0=(ones(input.dims.nx,input.dims.nz) .*vmax .*
(input.PML.nPML+1) .*log(1/input.PML.Rc)/2/input.PML.lp/input.dims.dx);
beta1=(@zeros(input.dims.nx,input.dims.nz));
beta3=copy(beta1);
tt=(1:input.PML.lp)/input.PML.lp;
tt2=repeat(reshape(tt,input.PML.lp,1),1,input.dims.nz);
plane_grad1=@zeros(input.dims.nx,input.dims.nz);
plane_grad3=copy(plane_grad1);

plane_grad1[2:input.PML.lp+1,:]=reverse(tt2,dims=1);
plane_grad1[nx-input.PML.lp:end-1,:]=tt2;
plane_grad1[1,:]=plane_grad1[2,:];
plane_grad1[end,:]=plane_grad1[end-1,:];

tt2=repeat(reshape(tt,1,input.PML.lp),nx,1);
plane_grad3[:,2:input.PML.lp+1]=reverse(tt2,dims=2);
plane_grad3[:,nz-input.PML.lp:end-1]=tt2;
plane_grad3[:,1]=plane_grad3[:,2];
plane_grad3[:,end]=plane_grad3[:,end-1];

beta1=beta0.*plane_grad1.^input.PML.nPML;
beta3=beta0.*plane_grad3.^input.PML.nPML;

plane_grad1=plane_grad3=vmax=nothing;

# receiver configuration
R=@zeros(input.dims.nt,length(input.receiver.r3));

# wave vector
u2=@zeros(input.dims.nx,input.dims.nz);
u2_last=@zeros(input.dims.nx,input.dims.nz);
u2_next=@zeros(input.dims.nx,input.dims.nz);
sigma23=@zeros(input.dims.nx,input.dims.nz);
sigma23_last=@zeros(input.dims.nx,input.dims.nz);
int_sigma23=@zeros(input.dims.nx,input.dims.nz);
int_sigma12=@zeros(input.dims.nx,input.dims.nz);
sigma12=@zeros(input.dims.nx,input.dims.nz);
sigma12_last=@zeros(input.dims.nx,input.dims.nz);
dsigma23=@zeros(input.dims.nx,input.dims.nz);
dsigma12=@zeros(input.dims.nx,input.dims.nz);
int_dsigma23=@zeros(input.dims.nx,input.dims.nz);
int_dsigma12=@zeros(input.dims.nx,input.dims.nz);
e23=@zeros(input.dims.nx,input.dims.nz);
e23_last=@zeros(input.dims.nx,input.dims.nz);
e12=@zeros(input.dims.nx,input.dims.nz);
e12_last=@zeros(input.dims.nx,input.dims.nz);
e23l_last=@zeros(input.dims.nx,input.dims.nz);
e12l_last=@zeros(input.dims.nx,input.dims.nz);

int_e23=@zeros(input.dims.nx,input.dims.nz);
int_e12=@zeros(input.dims.nx,input.dims.nz);
int_de23=@zeros(input.dims.nx,input.dims.nz);
int_de12=@zeros(input.dims.nx,input.dims.nz);


e23l=@zeros(input.dims.nx,input.dims.nz);
e12l=@zeros(input.dims.nx,input.dims.nz);
acc=@zeros(input.dims.nx,input.dims.nz);
f1=@zeros(input.dims.nx,input.dims.nz);
f2=@zeros(input.dims.nx,input.dims.nz);
de23=@zeros(input.dims.nx,input.dims.nz);
de12=@zeros(input.dims.nx,input.dims.nz);

sigma23_3_2_end=@zeros(input.dims.nx,input.dims.nz-1);
sigma12_1_2_end=@zeros(input.dims.nx-1,input.dims.nz);

Dx=[9/8/input.dims.dx -1/24/input.dims.dx];
Dz=[9/8/input.dims.dz -1/24/input.dims.dz];

pro_bar=Progress(input.dims.nt,1,"forward_simulation...",50);
# time stepping
for l=1:input.dims.nt-1
    @timeit ti "compute_strain_last" @parallel compute_strain_last(e23_last,e12_last,e23,e12);

    @timeit ti "compute_strain" @parallel compute_strain(input.dims.dx,input.dims.dz,e23,e12,e23_last,e12_last,Dx,Dz,u2);

    @timeit ti "compute_memory_variable_last" @parallel compute_memory_variable_last(e23l_last,e12l_last,e23l,e12l);
    @timeit ti "compute_memory_variable" @parallel compute_memory_variable(input.dims.dt,input.material_parameter.ts2,input.material_parameter.ts4,input.material_parameter.phi2,input.material_parameter.phi4,Dx,Dz,e23,e12,e23l,e12l,e23l_last,e12l_last);

    @timeit ti "compute_stress" @parallel compute_stress(input.dims.dt,input.material_parameter.C44,input.material_parameter.C46,input.material_parameter.C66,Dx,Dz,sigma23,sigma12,e23,e12,e23_last,e12_last,e23l,e12l,e23l_last,e12l_last,beta1,beta3,int_sigma23,int_sigma12);
    @timeit ti "shift coordinate" @parallel (2:input.dims.nx-1) z_2_end(sigma23,sigma23_3_2_end);
    @timeit ti "shift coordinate" @parallel (2:input.dims.nz-1) x_2_end(sigma12,sigma12_1_2_end);
    @timeit ti "compute_spatial_derivative_stress" @parallel compute_spatial_derivative_stress(input.dims.dt,input.dims.dx,input.dims.dz,Dx,Dz,sigma23_3_2_end,sigma12_1_2_end,dsigma23,dsigma12,int_dsigma23,int_dsigma12);
    @timeit ti "compute_acc" @parallel compute_acc(input.material_parameter.rho,acc,dsigma23,dsigma12,int_dsigma23,int_dsigma12,beta1,beta3);
    @timeit ti "compute_displacement" @parallel compute_displacement(input.dims.dt,Dx,Dz,u2,u2_last,u2_next,acc,beta1,beta3);

    if ns==1
        u2_next[CartesianIndex.(input.source.s1,input.source.s3)]=u2_next[CartesianIndex.(input.source.s1,input.source.s3)]+1 ./input.material_parameter.rho[CartesianIndex.(input.source.s1,input.source.s3)] .*input.source.src[l];
    else
        u2_next[CartesianIndex.(input.source.s1,input.source.s3)]=u2_next[CartesianIndex.(input.source.s1,input.source.s3)]+1 ./input.material_parameter.rho[CartesianIndex.(input.source.s1,input.source.s3)] .*input.source.src[l,:]';
    end
    u2_next[1,:]=u2_next[2,:];
    u2_next[:,1]=u2_next[:,2];
    @timeit ti "update_displacement" @parallel update_displacement(u2,u2_last,u2_next);

    # assign recordings
    @timeit ti "receiver" R[l+1,:]=reshape(u2_next[CartesianIndex.(input.receiver.r1,input.receiver.r3)],length(input.receiver.r3),);

    # plot
    if input.visualization.plot_interval!=0
        if mod(l,input.visualization.plot_interval)==0 || l==input.dims.nt-1
            vtkfile = vtk_grid(string(path_pic,"/wavefield_pic_",n_picture),input.dims.X,input.dims.Z);
            vtkfile["u2"]=u2_next;
            vtkfile["C44"]=input.material_parameter.C44;
            vtkfile["C66"]=input.material_parameter.C66;
            pvd[input.dims.dt*(l+1)]=vtkfile;
            n_picture=n_picture+1;
        end
    end

    next!(pro_bar);
end

data=R;
write2mat(string(path_rec,"/rec.mat"),data);

if input.visualization.plot_interval!=0
    vtk_save(pvd);
end

return u2_next,R
end

@parallel function compute_strain_last(e23_last,e12_last,e23,e12)
    @all(e23_last)=@all(e23);
    @all(e12_last)=@all(e12);
    return nothing;
end

@parallel function compute_strain(dx,dz,e23,e12,e23_last,e12_last,Dx,Dz,u2)
    @inn(e23)=@d_yi(u2)/dz;
    @inn(e12)=@d_xi(u2)/dx;
    return nothing;
end

@parallel function compute_memory_variable_last(e23l_last,e12l_last,e23l,e12l)
    @all(e23l_last)=@all(e23l);
    @all(e12l_last)=@all(e12l);
    return nothing
end

@parallel function compute_memory_variable(dt,ts2,ts4,phi2,phi4,Dx,Dz,e23,
    e12,e23l,e12l,e23l_last,e12l_last)
    @all(e23l)=.5 .*((2 .*dt .*@all(ts2) .*@all(phi2) .*@all(e23)+(2 .*@all(ts2) .-dt) .*@all(e23l)) ./(2 .*@all(ts2) .+dt)+@all(e23l_last));

    @all(e12l)=.5 .*((2 .*dt .*@all(ts4) .*@all(phi4) .*@all(e12)+(2 .*@all(ts4) .-dt) .*@all(e12l)) ./(2 .*@all(ts4) .+dt)+@all(e12l_last));
    return nothing
end

@parallel function compute_stress(dt,C44,C46,C66,Dx,Dz,sigma23,sigma12,e23,e12,e23_last,e12_last,e23l,e12l,e23l_last,e12l_last,beta1,beta3,int_sigma23,int_sigma12)
    @inn(sigma23)=dt .*(@inn(C44) .*(@inn(e23)-@inn(e23_last)) ./dt+
    @inn(C46) .*(@inn(e12)-@inn(e12_last)) ./dt+
    @inn(C44) .*(@inn(e23l)-@inn(e23l_last)) ./dt+
    @inn(C44) .*@inn(beta1) .*@inn(e23)+
    @inn(C46) .*@inn(beta3) .*@inn(e12)-
    (@inn(beta1)+@inn(beta3)) .*@inn(sigma23)-
    @inn(beta1) .*@inn(beta3) .*@inn(int_sigma23))+
    @inn(sigma23);

    @inn(sigma12)=dt*(@inn(C46) .*(@inn(e23)-@inn(e23_last)) ./dt+
    @inn(C66) .*(@inn(e12)-@inn(e12_last)) ./dt+
    @inn(C66) .*(@inn(e12l)-@inn(e12l_last)) ./dt+
    @inn(C46) .*@inn(beta1) .*@inn(e23)+
    @inn(C66) .*@inn(beta3) .*@inn(e12)-
    (@inn(beta1)+@inn(beta3)) .*@inn(sigma12)-
    @inn(beta1) .*@inn(beta3) .*@inn(int_sigma12))+
    @inn(sigma12);

    @inn(int_sigma23)=@inn(int_sigma23)+@inn(sigma23) .*dt;
    @inn(int_sigma12)=@inn(int_sigma12)+@inn(sigma12) .*dt;
    return nothing
end

@parallel function compute_spatial_derivative_stress(dt,dx,dz,Dx,Dz,sigma23_3_2_end,sigma12_1_2_end,dsigma23,dsigma12,int_dsigma23,int_dsigma12)
    @inn(dsigma23)=@d_yi(sigma23_3_2_end) ./dz;
    @inn(dsigma12)=@d_xi(sigma12_1_2_end) ./dx;

    @all(int_dsigma23)=@all(int_dsigma23)+@all(dsigma23) .*dt;
    @all(int_dsigma12)=@all(int_dsigma12)+@all(dsigma12) .*dt;
    return nothing
end

@parallel function compute_acc(rho,acc,dsigma23,dsigma12,int_dsigma23,int_dsigma12,beta1,beta3)
    @all(acc)=(@all(dsigma23)+@all(dsigma12)+@all(beta3) .*@all(int_dsigma12)+@all(beta1) .*@all(int_dsigma23))./@all(rho);
    return nothing
end

@parallel function compute_displacement(dt,Dx,Dz,u2,u2_last,u2_next,acc,beta1,beta3)
    @all(u2_next)=2 .*@all(u2)-@all(u2_last)+dt^2 .*@all(acc)-
    dt^2*@all(beta1).*@all(beta3) .*@all(u2)-
    dt*(@all(beta1)+@all(beta3)) .*(@all(u2)-@all(u2_last));
    return nothing
end

@parallel function update_displacement(u2,u2_last,u2_next)
    @all(u2_last)=@all(u2);
    @all(u2)=@all(u2_next);
    return nothing
end
end

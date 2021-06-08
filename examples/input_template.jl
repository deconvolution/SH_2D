## import packages
using SH_2D,SH_2D.SH_solver,ParallelStencil
Threads.nthreads()
## define input
mutable struct input2
    material_parameter
    dims
    PML
    source
    receiver
    visualization
end

mutable struct dims2
    nt
    nx
    nz
    X
    Z
    dt
    dx
    dz
end

mutable struct PML2
    lp
    nPML
    Rc
end

mutable struct material_parameter2
    C44
    C46
    C66
    ts2
    ts4
    phi2
    phi4
    rho
end

mutable struct source2
    s1
    s3
    s1t
    s3t
    src
end

mutable struct receiver2
    r1
    r3
    r1t
    r3t
end

mutable struct visualization2
    plot_interval
    path
end
## dimensions
dims=dims2(
1000, # nt
200, # nx
100, # nz
(1:1:200)*10, # X
(1:1:100)*10, # Z
.001, # dt
10, # dx
10); # dz
## PML
PML=PML2(20, # PML layers
2, # PML power, usually 2
.0001); # Theoretical reflection coefficient
## Material parameter
nx=dims.nx;
nz=dims.nz;
material_parameter=material_parameter2(zeros(nx,nz), # C44
zeros(nx,nz), # C46
zeros(nx,nz), # C66
zeros(nx,nz), # ts2
zeros(nx,nz), # ts4
zeros(nx,nz), # phi2
zeros(nx,nz), # phi4
zeros(nx,nz)); # rho

material_parameter.C44[:] .=10^9;
material_parameter.C46[:] .=-10^8;
material_parameter.C66[:] .=10^9;
material_parameter.ts2[:] .=.1;
material_parameter.ts4[:] .=.1;
material_parameter.phi2[:] .=-.1;
material_parameter.phi4[:] .=-.1;
material_parameter.rho[:] .=10^3;
## source

# magnitude
M=2.7;
# source frequency [Hz]
freq=10;
# source signal
singles=rickerWave(freq,dims.dt,dims.nt,M);

source=source2(50, # source grid location 1
50, # source grid location 3
500, # source true location 1
500, # source true location 3
singles);
## receiver
receiver=receiver2([30 40 60]', # receiver grid location 1
[30 30 30]', # receiver grid location 3
[300 400 600]', # receiver true location 1
[300 300 300]'); # receiver true location 3
## create folder for saving
p2= @__FILE__;
p3=chop(p2,head=0,tail=3);
if isdir(p3)==0
    mkdir(p3);
end
# plot
visualization=visualization2(100, # plot_interval
p3);
## summarize input file
input=input2(material_parameter,
dims,
PML,
source,
receiver,
visualization);
## pass input to solver mono_2D_sh()
u2_next,R=mono_2D_sh(input);

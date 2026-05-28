using SLMTools
using HDF5
using FileIO
using Images

# Generate an input grid and corresponding output grid.
Nx = 108
Ny = 108

L0 = natlat((Nx,Ny))
dL0 = dualShiftLattice(L0)

# Generate an input beam and target output beam.
inputBeam = lfGaussian(Intensity, L0, 1.0)
targetBeam = lfRect(Intensity, L0, (1.5,2.3))

# Use optimal transport to find an SLM phase to make an approximate output beam.
# phiOT = otPhase(inputBeam,targetBeam,0.001)

# Alternative optimal transport function.  This function is faster and works on larger 
# arrays, but is currently only implemented in 2 dimensions.  
# phiOT2 = otPhase2(inputBeam,targetBeam,0.0002,200)

# Refine the OT generated phase using the Gerchberg-Saxton algorithm.
# randGS = gs(inputBeam,targetBeam,10000,wrap(LF{RealPhase}(rand(size(targetBeam)...),targetBeam.L)))
# phiGS = gs(inputBeam,targetBeam,10000,phiOT)

# Try mraf
# phiMraf = mraf(sqrt(inputBeam),sqrt(targetBeam),10000,phiOT,CartesianIndices((30:80,40:70)),0.4)

# View the resulting output beams
# outputOT = square(sft(sqrt(inputBeam) * phiOT))
# outputOT2 = square(sft(sqrt(inputBeam) * phiOT2))
# outputrandGS = square(sft(sqrt(inputBeam) * randGS))
# outputGS = square(sft(sqrt(inputBeam) * phiGS))
outputMraf = square(sft(sqrt(inputBeam) * phiMraf)) 
# look(targetBeam,outputOT,outputOT2,outputGS,outputMraf)
look(inputBeam,targetBeam,outputOT,phiOT)
# look(inputBeam,targetBeam,outputOT2,phiOT2)
# look(inputBeam,targetBeam,outputrandGS,randGS)
# look(inputBeam,targetBeam,outputGS,phiGS)
look(inputBeam,targetBeam,outputMraf,phiMraf)

# save("inputBeam.png", Gray.(inputBeam.data))
# savePhase(phiOT, "OTprofile2.png")
# savePhase(phiMraf, "OTprofileMraf.png")
# h5write("OTprofile1.h5", "OT/inputBeam", inputBeam.data)
# h5write("OTprofile1.h5", "OT/targetBeam", targetBeam.data)
# h5write("OTprofile1.h5", "OT/phiOT", phiOT.data)
# h5write("OTprofile1.h5", "OT/outputOT", outputOT.data)
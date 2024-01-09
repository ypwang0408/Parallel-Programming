NVCC := nvcc
CXX := g++

SRCDIR = src
OBJDIR = obj
BINDIR = bin

NVCCFLAGS := -I./$(SRCDIR) -std=c++17 -gencode=arch=compute_86,code=sm_86 -Xcompiler -Wall -O3 
CXXFLAGS := -I./$(SRCDIR) -std=c++17 -Wall -O3

OPENMP_EXE := openmp.out
OPENMP_BLOCK_EXE := openmp_block.out
CUDA_EXE := cuda.out
CUDA2_EXE := cuda2.out
SEQUEN_EXE := sequential.out
CUDA_BLOCK_EXE := cuda_block

all: dirs cuda cuda2 cuda_block openmp openmp_block sequential 
.PHONY: clean all dirs

dirs:
		if [ ! -d "$(OBJDIR)" ]; then mkdir "$(OBJDIR)"; fi
		if [ ! -d "$(BINDIR)" ]; then mkdir "$(BINDIR)"; fi


cuda: $(OBJDIR)/util.o cuda.cu 
		$(NVCC) $^ -o $(BINDIR)/$(CUDA_EXE) $(NVCCFLAGS)

cuda2: $(OBJDIR)/util.o cuda2.cu 
		$(NVCC) $^ -o $(BINDIR)/$(CUDA2_EXE) $(NVCCFLAGS)

cuda_block: $(OBJDIR)/util.o cuda_block.cu 
		$(NVCC) $^ -o $(BINDIR)/$(CUDA_BLOCK_EXE)_2.out $(NVCCFLAGS) -DBLOCK_DIM=2
		$(NVCC) $^ -o $(BINDIR)/$(CUDA_BLOCK_EXE)_4.out $(NVCCFLAGS) -DBLOCK_DIM=4
		$(NVCC) $^ -o $(BINDIR)/$(CUDA_BLOCK_EXE)_8.out $(NVCCFLAGS) -DBLOCK_DIM=8
		$(NVCC) $^ -o $(BINDIR)/$(CUDA_BLOCK_EXE)_16.out $(NVCCFLAGS) 
		$(NVCC) $^ -o $(BINDIR)/$(CUDA_BLOCK_EXE)_24.out $(NVCCFLAGS) -DBLOCK_DIM=24
		$(NVCC) $^ -o $(BINDIR)/$(CUDA_BLOCK_EXE)_32.out $(NVCCFLAGS) -DBLOCK_DIM=32
		$(NVCC) $^ -o $(BINDIR)/$(CUDA_BLOCK_EXE)_48.out $(NVCCFLAGS) -DBLOCK_DIM=48

openmp: $(OBJDIR)/util.o openmp.cc 
		$(CXX) $^ -o $(BINDIR)/$(OPENMP_EXE) $(CXXFLAGS) -fopenmp

openmp_block: $(OBJDIR)/util.o openmp_block.cc 
		$(CXX) $^ -o $(BINDIR)/$(OPENMP_BLOCK_EXE) $(CXXFLAGS) -fopenmp


sequential: $(OBJDIR)/util.o sequential.cc
		$(CXX) $^ -o $(BINDIR)/$(SEQUEN_EXE) $(CXXFLAGS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cc
		$(CXX) -c $^ -o $@ $(CXXFLAGS)

clean:
	 	/bin/rm -rf $(BINDIR) $(OBJDIR)
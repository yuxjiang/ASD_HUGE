OS := $(shell uname)

PROJ=libcb
VER_MAJ=1
VER_MIN=0
CXX=g++

# Dependencies
ARMA_DIR=$(HOME)/Tools/arma # <replace with where armadillo is installed>

# Directories
INCDIR=../include
SRCDIR=../src
OBJDIR=../build
LIBDIR=../lib
BINDIR=../bin
EXAMPLE_DIR=../example

LIBS+=-L../lib # in case missing library, put them there
LIBS+=-lpthread
LIBS+=-lboost_system
LIBS+=-lboost_filesystem

CFLAGS+=-std=c++11
CFLAGS+=-O2
CFLAGS+=-Wall
CFLAGS+=-fdiagnostics-color=always
CFLAGS+=-I$(INCDIR)
CFLAGS+=-D ENABLE_LOGGING # for testing purposes
CFLAGS+=-fPIC
ifeq ($(OS),Darwin)
	CFLAGS+=-Xpreprocessor -fopenmp
	LIBS+=-lomp
else
	CFLAGS+=-fopenmp
	LIBS+=-lgomp
endif

ifneq ("$(wildcard /usr/local/include/armadillo)","")
	# armadillo installed on the system level
	LIBS+=-larmadillo
else ifeq ($(OS),Darwin)
	CFLAGS+=-I$(ARMA_DIR)/include
	CFLAGS+=-DARMA_DONT_USE_WRAPPER
	LIBS+=-framework Accelerate
else
	CFLAGS+=-I$(ARMA_DIR)/include
	CFLAGS+=-DARMA_DONT_USE_WRAPPER
	LIBS+=-lblas
	LIBS+=-llapack
endif

OBJS+=$(OBJDIR)/annotation.o
OBJS+=$(OBJDIR)/cafa.o
OBJS+=$(OBJDIR)/evaluation.o
OBJS+=$(OBJDIR)/fanngo.o
OBJS+=$(OBJDIR)/feature.o
OBJS+=$(OBJDIR)/gene_network.o
OBJS+=$(OBJDIR)/neural_network.o
OBJS+=$(OBJDIR)/ontology.o
OBJS+=$(OBJDIR)/sequence.o
OBJS+=$(OBJDIR)/util.o

SRCS+=$(SRCDIR)/annotation.cpp
SRCS+=$(SRCDIR)/cafa.cpp
SRCS+=$(SRCDIR)/evaluation.cpp
SRCS+=$(SRCDIR)/fanngo.cpp
SRCS+=$(SRCDIR)/feature.cpp
SRCS+=$(SRCDIR)/gene_network.cpp
SRCS+=$(SRCDIR)/neural_network.cpp
SRCS+=$(SRCDIR)/ontology.cpp
SRCS+=$(SRCDIR)/sequence.cpp
SRCS+=$(SRCDIR)/util.cpp

HEADERS+=$(INCDIR)/$(PROJ)/annotation.h
HEADERS+=$(INCDIR)/$(PROJ)/cafa.h
HEADERS+=$(INCDIR)/$(PROJ)/classification.h
HEADERS+=$(INCDIR)/$(PROJ)/data_matrix.h
HEADERS+=$(INCDIR)/$(PROJ)/distribution.h
HEADERS+=$(INCDIR)/$(PROJ)/evaluation.h
HEADERS+=$(INCDIR)/$(PROJ)/fanngo.h
HEADERS+=$(INCDIR)/$(PROJ)/feature.h
HEADERS+=$(INCDIR)/$(PROJ)/gene_network.h
HEADERS+=$(INCDIR)/$(PROJ)/graph.h
HEADERS+=$(INCDIR)/$(PROJ)/neural_network.h
HEADERS+=$(INCDIR)/$(PROJ)/ontology.h
HEADERS+=$(INCDIR)/$(PROJ)/sequence.h
HEADERS+=$(INCDIR)/$(PROJ)/util.h
HEADERS+=$(INCDIR)/csv.h

EXAMPLE+=$(BINDIR)/annotation.run
EXAMPLE+=$(BINDIR)/eval.run
EXAMPLE+=$(BINDIR)/fasta.run
EXAMPLE+=$(BINDIR)/fisher.run
EXAMPLE+=$(BINDIR)/nbp.run
EXAMPLE+=$(BINDIR)/nn.run
EXAMPLE+=$(BINDIR)/ont.run
EXAMPLE+=$(BINDIR)/quantile.run
EXAMPLE+=$(BINDIR)/test.run
EXAMPLE+=$(BINDIR)/subont.run

ifeq ($(OS),Darwin)
	NAMEOPT=-install_name
else
	NAMEOPT=-soname
endif

lib: dynamic static

all: dynamic static example

dynamic: $(LIBDIR)/$(PROJ).so

static: $(LIBDIR)/$(PROJ).a

example: $(EXAMPLE)

# dynamic library
$(LIBDIR)/$(PROJ).so: $(OBJS)
	$(CXX) -shared -Wl,$(NAMEOPT),$(PROJ).so.$(VER_MAJ)	-o $(LIBDIR)/$(PROJ).so.$(VER_MAJ).$(VER_MIN) $(OBJS) $(LIBS)
	ln -sf $(LIBDIR)/$(PROJ).so.$(VER_MAJ).$(VER_MIN) $(LIBDIR)/$(PROJ).so.$(VER_MAJ)
	ln -sf $(LIBDIR)/$(PROJ).so.$(VER_MAJ).$(VER_MIN) $(LIBDIR)/$(PROJ).so

# static library
$(LIBDIR)/$(PROJ).a: $(OBJS)
	ar rcs $@ $(OBJS)

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS)
	$(CXX) -c $(CFLAGS) -o $@ $<

# example (linking static library)
$(BINDIR)/%.run: $(EXAMPLE_DIR)/%_main.o $(LIBDIR)/$(PROJ).a
	$(CXX) -o $@ $^ $(LIBS)

$(EXAMPLE_DIR)/%_main.o: $(EXAMPLE_DIR)/%_main.cpp $(HEADERS)
	$(CXX) -c $(CFLAGS) -o $@ $<

.PHONY: clean

.PRECIOUS: $(OBJS)

clean:
	rm $(EXAMPLE) $(LIBDIR)/libcb* $(OBJDIR)/*.o

OS := $(shell uname)

CXX=g++
SRCDIR=../src
OBJDIR=../build
BINDIR=../bin

# Dependencies
ARMA_DIR=$(HOME)/Tools/arma # <where armadillo is installed>

LIBCB_INC=../../libcb-CPP/include
LIBCB=../../libcb-CPP/lib/libcb.a

LIBS+=$(LIBCB)
LIBS+=-lpthread
LIBS+=-L/usr/local/lib
LIBS+=-lboost_system
LIBS+=-lboost_filesystem

CFLAGS+=-std=c++11
CFLAGS+=-O2
CFLAGS+=-Wall
CFLAGS+=-fdiagnostics-color=always
CFLAGS+=-I $(LIBCB_INC)
CFLAGS+=-D ENABLE_LOGGING
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

TARGETS+=$(BINDIR)/ernet.run
TARGETS+=$(BINDIR)/get-auc.run
TARGETS+=$(BINDIR)/inspect-net.run
TARGETS+=$(BINDIR)/make-adj.run
TARGETS+=$(BINDIR)/make-gene-label.run
TARGETS+=$(BINDIR)/make-net.run
TARGETS+=$(BINDIR)/merge-net.run
TARGETS+=$(BINDIR)/nbp.run
TARGETS+=$(BINDIR)/print-net.run
TARGETS+=$(BINDIR)/tail-stat.run
TARGETS+=$(BINDIR)/align-score.run
TARGETS+=$(BINDIR)/neighbor-of.run

HEADERS+=$(LIBCB_INC)/libcb/util.h
HEADERS+=$(LIBCB_INC)/libcb/graph.h
HEADERS+=$(LIBCB_INC)/libcb/ontology.h
HEADERS+=$(LIBCB_INC)/libcb/annotation.h
HEADERS+=$(LIBCB_INC)/libcb/evaluation.h
HEADERS+=$(LIBCB_INC)/libcb/sequence.h
HEADERS+=$(LIBCB_INC)/libcb/gene_network.h
HEADERS+=$(LIBCB_INC)/csv.h

all: $(TARGETS)

$(BINDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS)
	$(CXX) -c $(CFLAGS) -o $@ $<

$(BINDIR)/%.run: $(OBJDIR)/%_main.o $(LIBCB)
	$(CXX) -o $@ $^ $(LIBS)

$(OBJDIR)/%_main.o: $(SRCDIR)/%_main.cpp $(SRCS) $(HEADERS)
	$(CXX) -c $(CFLAGS) -o $@ $<

.PHONY: clean

.PRECIOUS: $(OBJDIR)/%_main.o

clean:
	rm $(TARGETS) $(OBJDIR)/*.o

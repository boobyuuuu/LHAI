# Author: Zhiguo Yao <zhiguo.yao@ihep.ac.cn>, 2011/06/01
include Makefile.arch

#------------------------------------------------------------------------------

SHELL          = /bin/bash
DEBUG          = no

#MAINSRCS       = Src_Main.cc
#MAINSRCS       = Src_TSMap.cc
#MAINSRCS       = Src_GetSBP.cc
MAINSRCS       = Src_Convo_Template.cc 
MAINOBJS       = $(MAINSRCS:.cc=.o)
MAINEXES       = $(MAINOBJS:.o=)

CLASSSRCS      = 
CLASSOBJS      = $(CLASSSRCS:.cc=.o)
DICTOBJS       = $(CLASSOBJS:.o=Dict.o)
DICTSRCS       = $(DICTOBJS:.o=.cc)
LINKDEFS       = $(CLASSOBJS:.o=LinkDef.h)

MISCOBJS       = $(MISCSRCS:.cc=.o)

MISCSRCS       = src/basic/hpatimer.cc src/basic/papi.cc

PROGRAMS       = $(MAINEXES)
OBJS           = $(CLASSOBJS) $(DICTOBJS) $(MISCOBJS)
ALLOBJS        = $(MAINOBJS) $(OBJS)

LIBS          += -lMinuit -lz -L$(SLALIB_LIBDIR) -lsla -lyaml-cpp -L/home/lhaaso/hushicong/MyEnv/YAML_CPP/libs/yaml-cpp

ifeq ($(DEBUG),yes)
CXXFLAGS      := -g -DDEBUG $(filter-out -O -O1 -O2 -O3 -O4,$(CXXFLAGS))
LDFLAGS       := $(filter-out -O -O1 -O2 -O3 -O4,$(LDFLAGS))
endif

LDFLAGS       := -Wl,-rpath=/home/lhaaso/hushicong/MyEnv/YAML_CPP/libs/yaml-cpp

CXXFLAGS      += -I$(SLALIB_INCDIR)
CXXFLAGS      += -I/home/lhaaso/hushicong/MyEnv/YAML_CPP/include
CXXFLAGS      += -D_MAIN_

#------------------------------------------------------------------------------
.SUFFIXES: .cc .o .so .h .d
.PHONY: clean distclean test

all: $(PROGRAMS)

ifeq ($(findstring $(MAKECMDGOALS),clean distclean test),)
include $(ALLOBJS:.o=.d)
endif

test: $(PROGRAMS) exam.dat
	./$< exam.dat exam.root

clean:
	@rm -f $(OBJS) core *.so lib*.a *.d *.d.[0-9]* *.o *Dict.* *LinkDef.h *.pcm

distclean: clean
	@rm -f $(PROGRAMS) *.root *.eps *.dat

$(PROGRAMS): % : %.o $(OBJS)
	$(CXX) $(LDFLAGS) $^ $(LIBS) -o $@
	@echo "$@ done"

$(ALLOBJS): %.o : %.cc %.d
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(DICTSRCS): %Dict.cc: %.h %LinkDef.h
	@echo "Generating dictionary $@..."
	@rootcint -f $@ -c $^

$(LINKDEFS): %LinkDef.h: %.d
	@echo "#ifdef __CINT__" > $@; \
	echo "" >> $@; \
	echo "#pragma link off all globals;" >> $@; \
	echo "#pragma link off all classes;" >> $@; \
	echo "#pragma link off all functions;" >> $@; \
	echo "" >> $@; \
	echo "#pragma link C++ class $*+;" >> $@; \
	echo "" >> $@; \
	echo "#endif" >> $@

$(ALLOBJS:.o=.d): %.d: %.cc
	@set -e; rm -f $@; \
	$(CXX) -MM $(CXXFLAGS) $< > $@.$$$$; \
	DIR=$$(dirname $*)"/"; BASE=$$(basename $*); \
	[ "$$DIR" != "./" ] || DIR=""; \
	sed "s,\($$BASE\)\.o[ :]*,$$DIR\1.o $@: ,g" < $@.$$$$ > $@; \
	rm -f $@.$$$$


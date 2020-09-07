CXX=g++
RM=rm -f
# BOOST_ROOT=/usr/local/Cellar/boost/1.69.0_2/
# Load boost module to set environment variable BOOST_ROOT
INCLUDE=-I$(BOOST_ROOT)
CXXFLAGS=-std=c++11 -O2 -DNDEBUG $(INCLUDE)
LDFLAGS=

SRCS= \
      main.cpp \
      graph.cpp

OBJS=$(subst .cpp,.o,$(SRCS))

all: graph_transform

graph_transform: $(OBJS)
	$(CXX) $(LDFLAGS) -o $@ $(OBJS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CXX) $(CPPFLAGS) -MM $^ >> ./.depend;

clean:
	$(RM) $(OBJS) graph_transform

dist-clean: clean
	$(RM) *~ .dependtool

include .depend

CC=gcc
CXX=g++
RM=rm -f
# BOOST_ROOT=../../prototype/lib/boost_1_58_0/
INCLUDE=-I$(BOOST_ROOT)
CPPFLAGS=-std=c++11 $(INCLUDE) -DNDEBUG -O2
LDFLAGS=  
LDLIBS=

SRCS= \
      main.cpp

OBJS=$(subst .cpp,.o,$(SRCS))

all: network_analyzer

network_analyzer: $(OBJS)
	$(CXX) $(LDFLAGS) -o network_analyzer $(OBJS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	rm -f ./.depend
	$(CXX) $(CPPFLAGS) -MM $^ >> ./.depend;

clean:
	$(RM) $(OBJS) network_analyzer

dist-clean: clean
	$(RM) *~ .dependtool

include .depend

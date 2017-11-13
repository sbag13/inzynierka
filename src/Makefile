a.out: MRubyInterpreter.cpp MRubyInterpreter.hpp MRubyInterpreterTest.cpp
	$(CXX) MRubyInterpreter.cpp MRubyInterpreterTest.cpp Logger.cpp -L/home/korda/Desktop/microflow-0.1/third_party/mruby-1.3.0/build/host/lib -I/home/korda/Desktop/microflow-0.1/third_party/mruby-1.3.0/include -I/home/korda/Desktop/microflow-0.1/third_party/png++-0.2.5/ -lmruby -std=c++11 -g -o exec 

CXX = nvcc -Xcompiler="-fpermissive"
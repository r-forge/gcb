EXT := c

OBJS := ardblas.o

#compiler/preprocessor options
INCS := -I/usr/share/R/include -I/usr/local/cuda/include
PARAMS := -fpic -g

#linker options
LIBS := -L/usr/lib64/R/lib -lR -L/usr/local/cuda/lib64 -lcublas

CC := gcc -std=gnu99

TARGETS := libardblas.so

all: $(TARGETS) 

$(TARGETS): $(OBJS) $(COBJS)
	$(CC) -shared $(LD_PARAMS) $(LIBS) $(OBJS) -o $@

$(OBJS): %.o: %.$(EXT)
	$(CC) -c $(INCS) $(PARAMS) $^ -o $@
	
clean:
	rm -rf *o

.PHONY: all clean

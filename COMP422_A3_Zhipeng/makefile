CC=mpicc
CFLAGS= -mpilog
LDFLAGS= -lm
SOURCES=main.c matrixproduct.c
OBJECTS=
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE):
	$(CC) $(CFLAGS) $(SOURCES) -o $@ $(LDFLAGS)

clean:

	rm $(EXECUTABLE)


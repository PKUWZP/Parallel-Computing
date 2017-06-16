CC=gcc
CFLAGS=-pthread -g
LDFLAGS=-lm 
SOURCES=main.c gaussian_el.c final_checking.c Buildup.c back_sub.c
OBJECTS=
EXECUTABLE=main

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE):
	$(CC) $(CFLAGS) $(LDFLAGS) $(SOURCES) -o $@ 

clean:

	rm $(EXECUTABLE)


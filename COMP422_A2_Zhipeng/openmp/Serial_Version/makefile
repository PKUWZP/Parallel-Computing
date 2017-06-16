# This is the serial version of makefile


CC=xlc_r
CFLAGS=-qsmp=noomp
LDFLAGS=-lm
SOURCES=main.c gaussian_el.c final_checking.c back_sub.c
OBJECTS=
EXECUTABLE=main_serial

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE):
	$(CC) $(CFLAGS) $(LDFLAGS) $(SOURCES) -o $@ 

clean:

	rm $(EXECUTABLE)


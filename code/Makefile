# ==================================================================================== #
# = Folder structure
# ==================================================================================== #
SYS_INCDIR=/usr/local/include
SYS_LIBDIR=/usr/local/lib

EXEC   = deepflow
INCDIR = include
LIBDIR = lib
BINDIR = bin
SRCDIR = src

LDINCDIR = $(SYS_INCDIR) include
LDLIBDIR = $(SYS_LIBDIR) lib
INC_PARAMS=$(LDINCDIR:%=-I%) 
LIB_PARAMS=$(LDLIBDIR:%=-L%)

# ==================================================================================== #
# = Compiler settings
# ==================================================================================== #
CC      = gcc
CFLAGS  = -Wall -g -O3 -march=core-avx-i -std=c1x $(INC_PARAMS)
DFLAGS  = -g -Wall
LDFLAGS = -g -Wall -O3 -march=core-avx-i -std=c1x $(LIB_PARAMS)
LDLIBS  = -lm -ljpeg -lpng


# ==================================================================================== #
# = Targets
# ==================================================================================== #
TARGET = $(BINDIR)/$(EXEC)

LIBOBJ = $(BINDIR)/CPUFlags.o \
		$(BINDIR)/CPUInfo.o \
		$(BINDIR)/function.o \
		$(BINDIR)/performance.o \
		$(BINDIR)/perf.o \
		$(BINDIR)/io.o \
		$(BINDIR)/image.o \
		$(BINDIR)/opticalflow.o \
		$(BINDIR)/opticalflow_aux.o \
		$(BINDIR)/solver.o

SRCOBJ = $(BINDIR)/deepflow.o

# ==================================================================================== #
# = Build Commands
# ==================================================================================== #

$(BINDIR)/%.o : $(SRCDIR)/%.c $(SRCDIR)/%.h
	$(CC) -c $(DFLAGS) $(CFLAGS) $< -o $@

$(BINDIR)/%.o : $(SRCDIR)/%.c $(INCDIR)/%.h
	$(CC) -c $(DFLAGS) $(CFLAGS) $< -o $@

$(BINDIR)/%.o : $(LIBDIR)/%.c $(INCDIR)/%.h
	$(CC) -c $(CFLAGS) $< -o $@


all: $(LIBOBJ) $(SRCOBJ)
	$(CC) $(LIBOBJ) $(SRCOBJ) $(LDLIBS) -o $(TARGET) $(LDFLAGS)

clean:
	rm -rf $(LIBOBJ)
	rm -rf $(SRCOBJ)
	rm -rf $(TARGET)

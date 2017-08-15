include ../Makefile.defs

EXTRAFLAGS := -O0 -g

# Overwrite the -O3 flag that may have been set by Makefile.rules.
# This test cannot use it because ptxas does not support optimized debugging and compiling the test fails..
export FORCE_OPT=O0

################################################################################
#
# Include any special rule for this testcase here! Usually:
#
# static:
#
# dynamic:
#
################################################################################

include ../Makefile.rules


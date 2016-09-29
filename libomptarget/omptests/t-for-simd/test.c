#include <stdlib.h>
#include <stdio.h>

#include "../utilities/check.h"
#include "../utilities/utilities.h"

#define N 100

int main ()
{
  int a[N], b[N], c[N];

  check_offloading();

  long cpuExec = 0;
#pragma omp target map(tofrom: cpuExec)
  {
    cpuExec = omp_is_initial_device();
  }

  if (!cpuExec) {
  
    // Test: no clauses
    int fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: private, firstprivate, lastprivate, linear
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    int q = -5;
    int p = -3;
    int r = 0;
    int l = 10;
#pragma omp target map(tofrom: a, r) map(to: b,c)
#pragma omp parallel
#pragma omp for simd private(q) firstprivate(p) lastprivate(r) linear(l:2)
    for (int i = 0 ; i < N ; i++) {
      q = i + 5;
      p += i + 2;
      a[i] += p*b[i] + c[i]*q +l;
      r = i;
    }
    for (int i = 0 ; i < N ; i++) {
      int expected = (-1 + (-3 + i + 2)*i + (2*i)*(i + 5) + 10+(2*i));
      if (a[i] != expected) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], expected);
	fail = 1;
      }
    }
    if (r != N-1) {
      printf("Error for lastprivate: device = %d, host = %d\n", r, N-1);
      fail = 1;
    }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule static no chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(static)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule static no chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: static)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");  

    // Test: schedule static no chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: static)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");  

  
    // Test: schedule static chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    int ch = 10;
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(static, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule static chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: static, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule static chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: static, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule dyanmic no chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }
 
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(dynamic)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

// hangs
#if 0
    // Test: schedule dyanmic no chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }
 
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: dynamic)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule dyanmic no chunk, nonmonotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }
 
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(nonmonotonic: dynamic)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");
#endif

    // Test: schedule dyanmic no chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }
 
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: dynamic)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");
 
    // Test: schedule dynamic chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(dynamic, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

#if 0
    // Test: schedule dynamic chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: dynamic, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule dynamic chunk, nonmonotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(nonmonotonic: dynamic, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");
#endif

    // Test: schedule dynamic chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: dynamic, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");
  
    // Test: schedule guided no chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(guided)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

#if 0
    // Test: schedule guided no chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: guided)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule guided no chunk, nonmonotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(nonmonotonic: guided)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");
#endif

    // Test: schedule guided no chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: guided)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");
  
    // Test: schedule guided chunk
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(guided, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

#if 0
    // Test: schedule guided chunk, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: guided, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule guided chunk, nonmonotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(nonmonotonic: guided, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");
#endif

    // Test: schedule guided chunk, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

    ch = 10;
#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: guided, ch)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");
  
    // Test: schedule auto
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(auto)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule auto, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: auto)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule auto, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: auto)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: schedule runtime
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(runtime)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

#if 0
    // Test: schedule runtime, monotonic
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(monotonic: runtime)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");
#endif

    // Test: schedule runtime, simd
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd schedule(simd: runtime)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: collapse
    fail = 0;
    int ma[N][N], mb[N][N], mc[N][N];
    for (int i = 0 ; i < N ; i++)
      for (int j = 0 ; j < N ; j++) {
	ma[i][j] = -1;
	mb[i][j] = i;
	mc[i][j] = 2*i;
      }

#pragma omp target map(tofrom: ma) map(to: mb,mc)
#pragma omp parallel
#pragma omp for simd collapse(2)
    for (int i = 0 ; i < N ; i++)
      for (int j = 0 ; j < N ; j++)
	ma[i][j] += mb[i][j] + mc[i][j];

    for (int i = 0 ; i < N ; i++)
      for (int j = 0 ; j < N ; j++)
	if (ma[i][j] != (-1 + i + 2*i)) {
	  printf("Error at %d: device = %d, host = %d\n", i, ma[i][j], (-1 + i + 2*i));
	  fail = 1;
	}

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: ordered
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd ordered
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: nowait
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd nowait
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: safelen
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd safelen(16)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: simdlen
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd simdlen(16)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

    // Test: aligned
    fail = 0;
    for (int i = 0 ; i < N ; i++) {
      a[i] = -1;
      b[i] = i;
      c[i] = 2*i;
    }

#pragma omp target map(tofrom: a) map(to: b,c)
#pragma omp parallel
#pragma omp for simd aligned(a,b,c:8)
    for (int i = 0 ; i < N ; i++)
      a[i] += b[i] + c[i];

    for (int i = 0 ; i < N ; i++)
      if (a[i] != (-1 + i + 2*i)) {
	printf("Error at %d: device = %d, host = %d\n", i, a[i], (-1 + i + 2*i));
	fail = 1;
      }

    if (fail)
      printf ("Failed\n");
    else
      printf("Succeeded\n");

  } else {
    DUMP_SUCCESS(27);
  }

  
  return 0;
}

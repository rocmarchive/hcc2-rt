#include <stdio.h>
#define N 1024

#define TEST_SIMPLE     1

int a[N], b[N];

int main() {
  int i;
  int error, totError = 0;


#if TEST_SIMPLE
  for (i=0; i<N; i++) a[i] = b[i] = i;

  // alloc, move to
  #pragma omp target enter data nowait map(alloc: a[0:N/4])       map(to: b[0:N/4])
  #pragma omp target enter data nowait map(alloc: a[N/4:N/4])     map(to: b[N/4:N/4])
  #pragma omp target enter data nowait map(alloc: a[N/2:N/4])     map(to: b[N/2:N/4])
  #pragma omp target enter data nowait map(alloc: a[3*(N/4):N/4]) map(to: b[3*(N/4):N/4])
  #pragma omp taskwait

  // compute
  #pragma omp target nowait map(from: a[0:N/4])       map(to: b[0:N/4])
  {
    int j;
    for(j=0; j<N/4; j++) a[j] = b[j]+1;
  }
  #pragma omp target nowait  map(from: a[N/4:N/4])     map(to: b[N/4:N/4])
  {
    int j;
    for(j=N/4; j<N/2; j++) a[j] = b[j]+1;
  }
  #pragma omp target nowait map(from: a[N/2:N/4])     map(to: b[N/2:N/4])
  {
    int j;
    for(j=N/2; j<3*(N/4); j++) a[j] = b[j]+1;
  }
  #pragma omp target nowait map(from: a[3*(N/4):N/4]) map(to: b[3*(N/4):N/4])
  {
    int j;
    for(j=3*(N/4); j<N; j++) a[j] = b[j]+1;
  }
  #pragma omp taskwait


  #pragma omp target exit data nowait map(from: a[0:N/4])       map(release: b[0:N/4])
  #pragma omp target exit data nowait map(from: a[N/4:N/4])     map(release: b[N/4:N/4])
  #pragma omp target exit data nowait map(from: a[N/2:N/4])     map(release: b[N/2:N/4])
  #pragma omp target exit data nowait map(from: a[3*(N/4):N/4]) map(release: b[3*(N/4):N/4])
  #pragma omp taskwait


  error=0;
  for (i=0; i<N; i++) {
    if (a[i] != i+1) printf("%d: error %d != %d, error %d\n", i, a[i], i+1, ++error);
  }
  if (! error) {
    printf("  test with simple conpleted successfully\n");
  } else {
    printf("  test with simple conpleted with %d error(s)\n", error);
    totError++;
  }
#endif

  printf("completed with %d errors\n", totError);
  return totError;
}

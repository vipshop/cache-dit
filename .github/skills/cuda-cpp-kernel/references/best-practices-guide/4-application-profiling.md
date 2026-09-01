# 4. Application Profiling


##  4.1. Profile 

Many codes accomplish a significant portion of the work with a relatively small amount of code. Using a profiler, the developer can identify such hotspots and start to compile a list of candidates for parallelization.

###  4.1.1. Creating the Profile 

There are many possible approaches to profiling the code, but in all cases the objective is the same: to identify the function or functions in which the application is spending most of its execution time.

Note

**High Priority:** To maximize developer productivity, profile the application to determine hotspots and bottlenecks.

The most important consideration with any profiling activity is to ensure that the workload is realistic - i.e., that information gained from the test and decisions based upon that information are relevant to real data. Using unrealistic workloads can lead to sub-optimal results and wasted effort both by causing developers to optimize for unrealistic problem sizes and by causing developers to concentrate on the wrong functions.

There are a number of tools that can be used to generate the profile. The following example is based on `gprof`, which is an open-source profiler for Linux platforms from the GNU Binutils collection.
    
    
    $ gcc -O2 -g -pg myprog.c
    $ gprof ./a.out > profile.txt
    Each sample counts as 0.01 seconds.
      %   cumulative   self              self     total
     time   seconds   seconds    calls  ms/call  ms/call  name
     33.34      0.02     0.02     7208     0.00     0.00  genTimeStep
     16.67      0.03     0.01      240     0.04     0.12  calcStats
     16.67      0.04     0.01        8     1.25     1.25  calcSummaryData
     16.67      0.05     0.01        7     1.43     1.43  write
     16.67      0.06     0.01                             mcount
      0.00      0.06     0.00      236     0.00     0.00  tzset
      0.00      0.06     0.00      192     0.00     0.00  tolower
      0.00      0.06     0.00       47     0.00     0.00  strlen
      0.00      0.06     0.00       45     0.00     0.00  strchr
      0.00      0.06     0.00        1     0.00    50.00  main
      0.00      0.06     0.00        1     0.00     0.00  memcpy
      0.00      0.06     0.00        1     0.00    10.11  print
      0.00      0.06     0.00        1     0.00     0.00  profil
      0.00      0.06     0.00        1     0.00    50.00  report
    

###  4.1.2. Identifying Hotspots 

In the example above, we can clearly see that the function `genTimeStep()` takes one-third of the total running time of the application. This should be our first candidate function for parallelization. [Understanding Scaling](#understanding-scaling) discusses the potential benefit we might expect from such parallelization.

It is worth noting that several of the other functions in the above example also take up a significant portion of the overall running time, such as `calcStats()` and `calcSummaryData()`. Parallelizing these functions as well should increase our speedup potential. However, since APOD is a cyclical process, we might opt to parallelize these functions in a subsequent APOD pass, thereby limiting the scope of our work in any given pass to a smaller set of incremental changes.

###  4.1.3. Understanding Scaling 

The amount of performance benefit an application will realize by running on CUDA depends entirely on the extent to which it can be parallelized. Code that cannot be sufficiently parallelized should run on the host, unless doing so would result in excessive transfers between the host and the device.

Note

**High Priority:** To get the maximum benefit from CUDA, focus first on finding ways to parallelize sequential code.

By understanding how applications can scale it is possible to set expectations and plan an incremental parallelization strategy. [Strong Scaling and Amdahl’s Law](#strong-scaling-and-amdahls-law) describes strong scaling, which allows us to set an upper bound for the speedup with a fixed problem size. [Weak Scaling and Gustafson’s Law](#weak-scaling-and-gustafsons-law) describes weak scaling, where the speedup is attained by growing the problem size. In many applications, a combination of strong and weak scaling is desirable.

####  4.1.3.1. Strong Scaling and Amdahl’s Law 

Strong scaling is a measure of how, for a fixed overall problem size, the time to solution decreases as more processors are added to a system. An application that exhibits linear strong scaling has a speedup equal to the number of processors used.

Strong scaling is usually equated with Amdahl’s Law, which specifies the maximum speedup that can be expected by parallelizing portions of a serial program. Essentially, it states that the maximum speedup _S_ of a program is:

\\(S = \frac{1}{(1 - P) + \frac{P}{N}}\\)

Here _P_ is the fraction of the total serial execution time taken by the portion of code that can be parallelized and _N_ is the number of processors over which the parallel portion of the code runs.

The larger _N_ is(that is, the greater the number of processors), the smaller the _P/N_ fraction. It can be simpler to view _N_ as a very large number, which essentially transforms the equation into \\(S = 1/(1 - P)\\). Now, if 3/4 of the running time of a sequential program is parallelized, the maximum speedup over serial code is 1 / (1 - 3/4) = 4.

In reality, most applications do not exhibit perfectly linear strong scaling, even if they do exhibit some degree of strong scaling. For most purposes, the key point is that the larger the parallelizable portion _P_ is, the greater the potential speedup. Conversely, if _P_ is a small number (meaning that the application is not substantially parallelizable), increasing the number of processors _N_ does little to improve performance. Therefore, to get the largest speedup for a fixed problem size, it is worthwhile to spend effort on increasing _P_ , maximizing the amount of code that can be parallelized.

####  4.1.3.2. Weak Scaling and Gustafson’s Law 

Weak scaling is a measure of how the time to solution changes as more processors are added to a system with a fixed problem size _per processor_ ; i.e., where the overall problem size increases as the number of processors is increased.

Weak scaling is often equated with Gustafson’s Law, which states that in practice, the problem size scales with the number of processors. Because of this, the maximum speedup _S_ of a program is:

\\(S = N + (1 - P)(1 - N)\\)

Here _P_ is the fraction of the total serial execution time taken by the portion of code that can be parallelized and _N_ is the number of processors over which the parallel portion of the code runs.

Another way of looking at Gustafson’s Law is that it is not the problem size that remains constant as we scale up the system but rather the execution time. Note that Gustafson’s Law assumes that the ratio of serial to parallel execution remains constant, reflecting additional cost in setting up and handling the larger problem.

####  4.1.3.3. Applying Strong and Weak Scaling 

Understanding which type of scaling is most applicable to an application is an important part of estimating speedup. For some applications the problem size will remain constant and hence only strong scaling is applicable. An example would be modeling how two molecules interact with each other, where the molecule sizes are fixed.

For other applications, the problem size will grow to fill the available processors. Examples include modeling fluids or structures as meshes or grids and some Monte Carlo simulations, where increasing the problem size provides increased accuracy.

Having understood the application profile, the developer should understand how the problem size would change if the computational performance changes and then apply either Amdahl’s or Gustafson’s Law to determine an upper bound for the speedup.

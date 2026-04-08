# 7.12. cudaDevSmResourceGroupParams

**Source:** structcudaDevSmResourceGroupParams.html#structcudaDevSmResourceGroupParams


### Public Variables

unsigned int coscheduledSmCount

unsigned int flags

unsigned int preferredCoscheduledSmCount

unsigned int reserved[12]

unsigned int smCount


### Variables

unsigned int cudaDevSmResourceGroupParams::coscheduledSmCount


The amount of co-scheduled SMs grouped together for locality purposes.

unsigned int cudaDevSmResourceGroupParams::flags


Combination of `cudaDevSmResourceGroup_flags` values to indicate this this group is created.

unsigned int cudaDevSmResourceGroupParams::preferredCoscheduledSmCount


When possible, combine co-scheduled groups together into larger groups of this size.

unsigned int cudaDevSmResourceGroupParams::reserved[12]


Reserved for future use - ensure this is is zero initialized.

unsigned int cudaDevSmResourceGroupParams::smCount


The amount of SMs available in this resource.

* * *
